
from settings import *
import pandas as pd 
import abc
import argparse
args = abc.abstractproperty()


def process(data):
    csv_file = f'{data}.csv'
    data = pd.read_csv(os.path.join(DATA_RAW,csv_file))
    processed_data = data.drop(['car_01'], axis=1)
    processed_data.to_csv(os.path.join(DATA_PROCESSED,csv_file))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model training scripts')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    global_args = parse_args()
    args.train = global_args.train
    args.test = global_args.test

    if args.train:
        process('Train')
    if args.test:
        process('Test')


 











