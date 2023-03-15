import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = os.path.join(ROOT_DIR,"data")



DATA_RAW = os.path.join(DATA_DIR,"raw")

DATA_PROCESSED = os.path.join(DATA_DIR,"processed")

DATA_PREDICTED = os.path.join(DATA_DIR,"predictions")

MODEL_DIR = os.path.join(ROOT_DIR, "models")



TARGET_VARIABLE = "target"
DATASET_NAME = "creditcard.csv"


