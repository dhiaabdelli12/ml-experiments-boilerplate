import abc
import argparse
from xgboost import XGBRegressor
from settings import *
import pandas as pd 
from sklearn.model_selection import train_test_split
from  sklearn.metrics import mean_squared_error
import numpy as np 
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
import json
warnings.filterwarnings("ignore")
from process import *
from imblearn.over_sampling import ADASYN
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


args = abc.abstractproperty()

models = [
    RandomForestClassifier
]

def hyperparameter_tuning(space):

    data = pd.read_csv(os.path.join(DATA_PROCESSED, 'Train.csv'))
    feats = data.columns.tolist()
    feats.remove(TARGET_VARIABLE)
    X = data[feats]
    y = data[TARGET_VARIABLE]


    ada = ADASYN(random_state=42)
    X, y = ada.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = xgb.XGBClassifier(n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                         reg_alpha = int(space['reg_alpha']),min_child_weight=space['min_child_weight'],
                         colsample_bytree=space['colsample_bytree'])
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    model.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="rmse",
            early_stopping_rounds=10,verbose=False)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    #change the metric if you like
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}


def hyperopt():
    space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 1000,
        'learning_rate':0.01
    }

    
    trials = Trials()
    best = fmin(fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
    print(best)



def train(save, predict,grid):
    process('Train')
    data = pd.read_csv(os.path.join(DATA_PROCESSED, 'Train.csv'))
    feats = data.columns.tolist()
    feats.remove(TARGET_VARIABLE)
    X = data[feats]
    y = data[TARGET_VARIABLE]

    res={}

    scaler = StandardScaler()


    #ada = ADASYN(random_state=42)
    #X, y = ada.fit_resample(X, y)
    #res['ada_resam']='yes'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



    if grid== True:
        #hyperopt()
        model = XGBClassifier()
        #model = RandomForestClassifier()
     
        grid_search = GridSearchCV(
        model, 
        XGB_GRID_PARAMS, 
        cv=5, 
        n_jobs=-1, 
        scoring='accuracy',
        verbose=2 
        )

        # fit the GridSearchCV object to the data
        grid_search.fit(X, y)

        # print the best hyperparameters and their score
        print(grid_search.best_params_)
        print(grid_search.best_score_)
        exit()



    model = XGBClassifier(**XGB_PARAMS)

    
    
    model.fit(X=X_train,y=y_train)
    y_pred = model.predict(X_test)


    importance = model.feature_importances_

    df_importance = pd.DataFrame({'feature': feats, 'importance': importance})
    df_importance = df_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    print(df_importance)


    error_analysis(X_test,y_test,y_pred,predict[0])

    if predict is not None:
        process('Test')
        test = pd.read_csv(os.path.join(DATA_PROCESSED,'Test.csv'))
        feats = test.columns.tolist()
        feats.remove('ID')
        test['Target']  = model.predict(test[feats]) 
        sub = test[['ID','Target']]
        sub.to_csv(os.path.join(DATA_PREDICTED,f'{predict[0]}.csv'), index= False)


    
    res['model']= type(model).__name__,
    res['Accuracy']= accuracy_score(y_pred=y_pred,y_true=y_test),
    res['F1-score']= f1_score(y_pred=y_pred,y_true=y_test),
    res['features']=feats,
    res['hyperparameters']=XGB_PARAMS

    

    log(res)
    




def parse_args():
    parser = argparse.ArgumentParser(
        description='Model training scripts')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--predict', action='store', nargs=1, type=str)
    parser.add_argument('--log-corr', action='store_true')
    parser.add_argument('--hyperopt', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
  

    global_args = parse_args()
    args.save_model = global_args.save_model
    args.predict = global_args.predict
    args.hyperopt = global_args.hyperopt


    train(args.save_model, args.predict,args.hyperopt)

    

