import time
import torch
from datasets.BCI_dataset.dataloader import get_bci_dataloaders
from utils import Metric, TimeMetric, MetricGroup
import utils.args as args
from utils.args import get_model, get_optimizer
from utils.save import load_checkpoint, save_checkpoint


def inference(dataloader, model):
    model.eval()
    result = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['HE'].to(args.device)
            logits = model(inputs)
            result.append(logits.argmax(1))
    result = torch.cat(result, 0).cpu().numpy()
    return result


def validation(dataloader, model, criterion):
    metrics = MetricGroup([Metric('valid_avg_loss', 0.0), Metric('valid_avg_acc', 0.0)])
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['HE'].to(args.device)
            targets = batch['label'].to(args.device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            top1_preds = logits.argmax(1)
            top1_acc = torch.sum(top1_preds == targets)/targets.size(0)
            metrics.step([loss.detach(), top1_acc])
    return metrics.avg


def train(dataloader, model, criterion, optimizer, regularization=None):
    metrics = MetricGroup([Metric('train_avg_loss', 0.0), Metric('train_avg_acc', 0.0)]) # trace loss/acc of every batch
    metrics_log = MetricGroup([Metric('train_log_loss', 0.0), Metric('train_log_acc', 0.0)]) # trace loss/acc for each log_freq

    model.train()
    len_epoch = len(dataloader)
    for step, batch in enumerate(dataloader):
        inputs = batch['HE'].to(args.device)
        targets = batch['label'].to(args.device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if regularization is not None:
            reg = regularization(model)
            loss = loss + args.lamb * reg

        # update model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to reproduce
        optimizer.step()
        _, preds = logits.max(1)

        # statistics
        metrics.step([loss.detach(), torch.sum(preds == targets.data)/targets.size(0)])
        metrics_log.step([loss.detach(), torch.sum(preds == targets.data)/targets.size(0)])

        # Log during an epoch
        if (step+1) % args.log_freq == 0:
            print(f"\tStep {step+1}/{len_epoch}: {metrics_log}")
            metrics_log.reset()
    return metrics.avg
    

def finetune_classification():
    # 1) Dataset Load
    dataloaders = get_bci_dataloaders(args.dataset, batch_size=32, num_workers=4, image_size=args.image_size, aug_level=args.aug_level)
    num_classes = len(dataloaders['train'].dataset.HER2_LEVELS)
    
    # 2) Model Load, Loss Function, Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    model = get_model(num_classes)
    
    optimizer = get_optimizer(model)

    # 3) Load Checkpoint
    model, optimizer, last_epoch = load_checkpoint(model, optimizer)
    model.to(args.device)
    
    # 4) Variables for Training
    start_epoch, num_epochs = last_epoch + 1, 100
    best_acc, best_epoch = 0.0, 0
    best_acc5 = 0.0
    train_time = TimeMetric("Training Time", time.time())
    epoch_time = TimeMetric("Epoch Training Time", time.time())
    log = {}
    
    # 5) Train!
    for epoch in range(start_epoch, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}\n"+"-"*10+"\n",end="")
        epoch_time.reset() 

        # Train and Validation
        train_avg_loss, train_avg_acc = train(dataloaders['train'], model, criterion, optimizer)
        val_avg_loss, val_avg_acc = validation(dataloaders['val'], model, criterion)
        best_acc = max(best_acc, val_avg_acc)
        best_epoch = epoch if best_acc == val_avg_acc else best_epoch

        # Log
        print(f"Train Loss: {train_avg_loss:.4f}, Train Acc: {train_avg_acc:.4f}, Val Loss: {val_avg_loss:.4f}, Val Acc: {val_avg_acc:.4f}, Best Acc: {best_acc:.4f}, Best Epoch: {best_epoch} \n{epoch_time}")
        log = {"train_loss": train_avg_loss, 
                "train_accuracy": train_avg_acc, 
                "val_loss": val_avg_loss,
                "val_accuracy": val_avg_acc,
            }
        
        # save model
        if args.save_dir is not None: 
            args.save_dir = save_checkpoint(epoch, model, optimizer)

        # Early stopping
        stop_threshold = 10
        if epoch - best_epoch > stop_threshold:
            print("\nEarly Stopping ...\n")
            break
            
    # 6) Log after training
    time_elapsed = train_time.elapsed()
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best Validation Accuracy: {}, Epoch: {}".format(best_acc, best_epoch))

    return dataloaders, model