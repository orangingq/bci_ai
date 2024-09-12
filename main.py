import args
from utils import random_seed

def main():
    args.set_args()
    random_seed(args.seed)


    # 1) Dataset Load
    dataloaders, _, args.num_classes, args.image_size = get_dataloaders(args.dataset, batch_size=args.batch_size, num_workers=4, image_size=224, distributed=(args.dp > 1))
    args.num_batches = len(dataloaders['train'])

    pass

if __name__ == '__main__':
    main()