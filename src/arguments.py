import argparse
from typing import Any


def parse_args(custom_args:dict=None):
    """Parse arguments.

    Args:
        custom_args:
    """
    # Parameters settings
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

    # Dataset setting
    # parser.add_argument('--dataroot', type=str, default="../data/东线环切数据_DJ_5.csv", help='path to dataset')
    parser.add_argument('--object_col', type=str, default="DJ_5", help='object column in the dataset')
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size [128]')
    parser.add_argument('--targetdomain', type=str, default="HZW", help='input batch size [128]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nums_hidden', type=int, default=32,
                        help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--LAMBDA', type=float, default=0.4,
                        help='判别损失权重')
    parser.add_argument('--BETA', type=float, default=0.4,
                        help='目标域回归器权重')
    # parser.add_argument('--nhidden_decoder', type=int, default=16,
    #                     help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=10, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=4e-3,
                        help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
    parser.add_argument('--seed', type=int, default=2028, help='random seed to use. Default=100')
    parser.add_argument('--debug', type=bool, default=False, help='whether in debug mode or not')

    # Parse arguments
    if custom_args is None:
        args = parser.parse_args()  # Use command-line arguments
    else:
        # Use a dictionary to merge defaults and custom arguments
        args = parser.parse_args([])
        default_args = vars(args)
        # custom_args_dict = vars(parser.parse_args(custom_args))
        default_args.update({k: v for k, v in custom_args.items() if v is not None})
        args = argparse.Namespace(**default_args)

    return args


if __name__ == '__main__':
    # 测试一下
    custom_args = {"dataroot": "../data/dataset1.csv", "object_col": "column1"}
    args = parse_args(custom_args)
    print(args)
    print(args.dataroot)
    print(args.object_col)
    print(args.batchsize)
