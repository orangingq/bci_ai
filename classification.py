import time
import torch
from BCI_dataset.dataloader import get_bci_dataloaders
from utils import Metric, TimeMetric, MetricGroup
import utils.args as args
from utils.args import get_model, get_optimizer
from utils.save import load_checkpoint, save_checkpoint


def inference(dataloader, model):
    model.eval()
    result = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['HE'].cuda()
            logits = model(inputs)
            result.append(logits.argmax(1))
    result = torch.cat(result, 0).cpu().numpy()
    return result


def validation(dataloader, model, criterion):
    metrics = MetricGroup([Metric('valid_avg_loss', 0.0), Metric('valid_avg_acc1', 0.0), Metric('valid_avg_acc5', 0.0)])
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['HE'].cuda()
            targets = batch['label'].cuda()
            logits = model(inputs)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            _, top5_preds = logits.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            top1_preds = top5_preds[0]
            top1_acc = torch.sum(top1_preds == targets)/targets.size(0)
            top5_acc = torch.sum(torch.sum(top5_preds == targets, dim=0, dtype=torch.bool))/targets.size(0)
            metrics.step([loss.detach(), top1_acc, top5_acc])
    return metrics.avg


def train(dataloader, model, criterion, optimizer, regularization=None):
    metrics = MetricGroup([Metric('train_avg_loss', 0.0), Metric('train_avg_acc', 0.0)]) # trace loss/acc of every batch
    metrics_log = MetricGroup([Metric('train_log_loss', 0.0), Metric('train_log_acc', 0.0)]) # trace loss/acc for each log_freq

    model.train()
    len_epoch = len(dataloader)
    for step, batch in enumerate(dataloader):
        inputs = batch['HE'].cuda()
        targets = batch['label'].cuda()
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
    dataloaders = get_bci_dataloaders(args.dataset, batch_size=32, num_workers=4, image_size=args.image_size)

    # 2) Model Load, Loss Function, Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    model = get_model()
    
    optimizer = get_optimizer(model)

    # 3) Load Checkpoint
    model, optimizer, last_epoch = load_checkpoint(model, optimizer)
    model.cuda()
    
    # 4) Variables for Training
    start_epoch, num_epochs = last_epoch + 1, 300
    best_acc1, best_epoch1 = 0.0, 0
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
        val_avg_loss, val_avg_acc1, val_avg_acc5 = validation(dataloaders['val'], model, criterion)
        best_acc1, best_acc5 = max(best_acc1, val_avg_acc1), max(best_acc5, val_avg_acc5)
        best_epoch1 = epoch if best_acc1 == val_avg_acc1 else best_epoch1

        # Log
        print(f"Train Loss: {train_avg_loss:.4f}, Train Acc: {train_avg_acc:.4f}, Val Loss: {val_avg_loss:.4f}, Val Top1 Acc: {val_avg_acc1:.4f}, Val Top5 Acc: {val_avg_acc5:.4f}, Best Acc: {best_acc1:.4f}, Best Epoch: {best_epoch1} \n{epoch_time}")
        log = {"train_loss": train_avg_loss, 
                "train_Top1_accuracy": train_avg_acc, 
                "val_loss": val_avg_loss,
                "val_Top1_accuracy": val_avg_acc1,
                "val_Top5_accuracy": val_avg_acc5,
            }
        
        # save model
        if args.save_dir is not None: 
            args.save_dir = save_checkpoint(epoch, model, optimizer)

        # Early stopping
        stop_threshold = 10
        if epoch - best_epoch1 > stop_threshold:
            print("\nEarly Stopping ...\n")
            break
            
    # 6) Log after training
    time_elapsed = train_time.elapsed()
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best Validation Accuracy: {}, Epoch: {}".format(best_acc1, best_epoch1))

    return dataloaders, model