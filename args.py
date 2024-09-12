
import argparse

seed = 42
dataset = 'cifar-10'
model_name = 'mixer'

def get_args():
    global_variables = [
        'seed', 
        'dataset',
        'model_name'
    ]
    return {var: globals()[var] for var in global_variables}

def set_args():
    global seed, dataset, model_name
    
    parser = argparse.ArgumentParser()
    # Arguments
    parser.add_argument('--seed', type=int, default=seed, help='Random Seed')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset')
    # model arguments
    parser.add_argument('--model_name', type=str, default=model_name, help='Model Name')
    args = parser.parse_args()
    for arg in vars(args):
        globals()[arg] = getattr(args, arg)# update global variables
    return
