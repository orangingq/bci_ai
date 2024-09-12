import torch
import utils.args as args

def save_checkpoint(epoch, model, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, args.save_dir)
    
    print(f"Model saved at {args.save_dir}")
    return 

def load_checkpoint(model, optimizer):
    if args.load_dir is None:
        return model, optimizer, 0
    checkpoint = torch.load(args.load_dir, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print(f"Model loaded from {args.load_dir}")
    return model, optimizer, epoch