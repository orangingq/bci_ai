import time
import torch
import utils.args as args
from utils.args import get_model, get_optimizer
from train import train, validation
from BCI_dataset.dataloader import get_bci_dataloaders
from utils import random_seed, load_checkpoint, save_checkpoint, TimeMetric

def main():
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

    return


if __name__ == '__main__':
    args.set_args()
    random_seed(args.seed)
    main()