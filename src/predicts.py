import argparse
import sys
from src.arguments import parse_args

if __name__ == '__main__':
    # from predict import predict # 单源域
    from predict import predict # 多源域
    parser = argparse.ArgumentParser()

    parser.add_argument('--targetdomain', required=True, help='Path to the data root')
    parser.add_argument('--object_col', required=True, help='Name of the object column')
    args = parser.parse_args()

    # 这里您可以直接使用 args.dataroot 和 args.object_col
    custom_args = {"targetdomain": args.targetdomain, "object_col": args.object_col}
    args = parse_args(custom_args)
    predict(args)

