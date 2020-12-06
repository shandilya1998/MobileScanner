from src.object_detection import train
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Parse training arguments')

parser.add_argument(
    '--gpu',
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
    help='use gpu or not'
)

parser.add_argument(
    '--net', 
    type=str, 
    required=True, 
    help='net type'
)

parser.add_argument(
    '--plot', 
    type=str2bool, 
    const=True, 
    nargs='?',
    default=False, 
    help='plot model choice'
)

parser.add_argument(
    '--epochs', 
    type=int, 
    default=120, 
    help='number of epochs of training'
)

parser.add_argument(
    '--train-batch-size',
    type=int,
    default=10,
    help='batch size for dataloader'
)

parser.add_argument(
    '--val-batch-size',
    type=int,
    default=10,
    help='batch size for dataloader'
)

parser.add_argument(
    '--eps', 
    type=float, 
    default=1e-8, 
    help='numerical stability factor for adam optimizer'
)   

parser.add_argument(
    '--beta1', 
    type=float, 
    default=0.9, 
    help='running average coefficient of gradient for adam optimizer'
)   

parser.add_argument(
    '--beta2', 
    type=float, 
    default=0.999, 
    help='running average coefficient of gradient square for adam optimizer'    )

parser.add_argument(
    '--job-dir',
    help='path for saving images in gcs'
)

parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='random manual seed'
)

parser.add_argument(
    '--log-interval',
    type=int,
    default=20,
    help='wandb log interval'
)

args = parser.parse_args()

train.run(
    log_interval = args.log_interval,
    job_dir = args.job_dir,
    train_batch_size = args.train_batch_size,
    val_batch_size = args.val_batch_size,
    learning_rate = args.learning_rate,
    eps = args.eps,
    beta1 = args.beta1,
    beta2 = args.beta2,
    architecture = args.net, 
    plot_model = args.plot,
    seed = args.seed
)
