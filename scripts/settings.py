import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = os.path.join(ROOT_DIR, "data")


DATA_RAW = os.path.join(DATA_DIR, "raw")

DATA_PROCESSED = os.path.join(DATA_DIR, "processed")

DATA_PREDICTED = os.path.join(DATA_DIR, "predictions")

MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")


TARGET_VARIABLE = 'Label'


FEATURES = [
    'I/O Data Operations', ' I/O Data Bytes',
       'Number of subprocesses', 'Time on processor', 'Disk Reading/sec',
       'Disc Writing/sec', 'Bytes Sent/sent', 'Received Bytes (HTTP)',
       'Network packets sent', 'Network packets received', 'Pages Read/sec',
       'Pages Input/sec', 'Page Errors/sec', 'Confirmed byte radius'
]


XGB_GRID_PARAMS = {
    'learning_rate': [0.5],
    'max_depth': [7],
    'n_estimators': [200,300,400],
    'subsample': [1.0],
    'colsample_bytree': [0.5],
    'gamma': [0.1]
}

XGB_PARAMS = {'colsample_bytree': 0.5, 'gamma': 0.1, 'learning_rate': 0.5, 'max_depth': 7, 'n_estimators': 300, 'subsample': 1.0}



CB_PARAMS = {
    'n_estimators' : 20000
}

RF_PARAMS = {'max_depth': 20, 'max_features': 'sqrt', 'n_estimators': 200}
