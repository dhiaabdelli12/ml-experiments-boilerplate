import abc
import argparse
args = abc.abstractproperty()



def parse_args():
    parser = argparse.ArgumentParser(
        description='Model training scripts')
    parser.add_argument('--save-model', action='store_true')

if __name__ == "__main__":
    global_args = parse_args()
    args.save_model = global_args.save_model

    if args.save_model == True:
        #function to save the model
        ...
