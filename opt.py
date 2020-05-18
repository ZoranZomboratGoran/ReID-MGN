import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="Market-1501-v15.09.15",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'visualize'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--weight',
                    default='weights/model.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=400,
                    type=int,
                    help='number of epoch to train')

parser.add_argument('--resume_epoch',
                    default=-1,
                    type=int,
                    help='epoch from which to resume training')

parser.add_argument('--lr',
                    default=2e-4,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[320, 380],
                    help='MultiStepLR,decay the learning rate')

parser.add_argument("--batchid",
                    default=4,
                    type=int,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    type=int,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=10,
                    type=int,
                    help='the batch size for test')

parser.add_argument("--savefreq",
                    default=10,
                    type=int,
                    help='the epoch frequency of saving weights')

parser.add_argument("--evalfreq",
                    default=10,
                    type=int,
                    help='the epoch frequency of evaluating model precision')

parser.add_argument("--usecpu",
                    default=False,
                    type=bool,
                    help='run train and evaluation on cpu')

parser.add_argument("--log_dir",
                    default="log",
                    type=str,
                    help='directotry for tensorboard logging')

opt = parser.parse_args()
