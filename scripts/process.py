
import pandas as pd
import json
from settings import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler


def log(res):
    filepath = os.path.join(LOG_DIR, 'logs.json')
    with open(filepath, "r+") as file:
        data = json.load(file)
        data.append(res)
        file.seek(0)
        json.dump(data, file)


def process(filename):
    csv_file = f'{filename}.csv'
    data = pd.read_csv(os.path.join(DATA_RAW, csv_file))

    features = FEATURES

    data['pack_byte_ratio'] = data['Network packets received'] / data['Received Bytes (HTTP)']
    data['sub_err_ratio'] = data['Number of subprocesses'] / data['Page Errors/sec']

    data['time_proc_0'] = 1
    data.loc[data['Time on processor'] == 0, 'time_proc_0'] = 0

    data['sent_rec_sum'] = data['Bytes Sent/sent'] + data['Received Bytes (HTTP)']

    engineered = [
        'pack_byte_ratio',
        'sub_err_ratio',
        'time_proc_0',
        'sent_rec_sum'
    ]



    if filename=='Train':
        processed_data = data[FEATURES + engineered+ [TARGET_VARIABLE]]
    else:
        processed_data = data[FEATURES + engineered+ ['ID']]

    processed_data.to_csv(os.path.join(DATA_PROCESSED, csv_file), index=False)


def error_analysis(X_test, y_test, y_pred, filename):
    wrong_indices = [i for i, (x, y) in enumerate(
        zip(y_test, y_pred)) if x != y]
    mask = X_test.index.isin(wrong_indices)
    result = X_test[mask]
    result.to_csv(os.path.join(DATA_PREDICTED,f'{filename}_wrong_predictions.csv'), index=False)
