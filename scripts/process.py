from imports import *

def clean() -> None:
    df = pd.read_csv(os.path.join(s.DATA_RAW, "train.csv"))
    df.dropna()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        default="processed",
        help="new processed data csv file name",
    )

    df = df[['humidity','temp_mean','pm2_5']]

    args = parser.parse_args()
    processed_data_filename = os.path.join(s.DATA_PROCESSED, args.name) +".csv"

    df.to_csv(processed_data_filename, encoding='utf-8', index=False)


if __name__ == "__main__":
    clean()
