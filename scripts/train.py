from statistics import mode
from imports import *
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from functools import wraps
from contextlib import contextmanager
from sklearn.metrics import classification_report
from datetime import datetime
import json


@contextmanager
def write_metrics(*args, **kwargs):
    with open(*args, **kwargs) as f:
        yield f


def model_registry(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        score, prediction, fmodel, model_name = function(*args, **kwargs)

        with write_metrics(os.path.join(s.MODEL_DIR, "logs", "{}_report.txt".format(datetime.today().strftime('%Y-%m-%d'))), mode='a+') as f:
            f.write("\n"+model_name + "\nScore: " + str(score) +
                    "\nParams: "+str(fmodel.get_xgb_params()))

        if serialize == 'y':
            serialize_model(fmodel,model_name+".json")

        return score, prediction, fmodel
    return wrapped


@model_registry
def fitter(model_name, model, X_train, X_test, y_train, y_test):
    md = model()
    md.fit(X_train, y_train)
    y_pred = md.predict(X_test)
    return np.sqrt((MSE(y_test, y_pred))), y_pred, md, model_name


def serialize_model(model,model_name) -> None:
    model.save_model(os.path.join(s.MODEL_DIR,"serialized", model_name))


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelname",
        default="model",
    )
    parser.add_argument(
        "--serialize",
        default="y",
    )

    args = parser.parse_args()

    dataset_loc = os.path.join(s.DATA_PROCESSED, "processed.csv")

    df = pd.read_csv(dataset_loc)
    y = df.pop(s.TARGET_VARIABLE)
    X = df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    global serialize
    serialize = args.serialize

    fitter(args.modelname, XGBRegressor, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train()
