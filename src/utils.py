import pandas as pd
import ast
import numpy as np
import pickle
import random
from src.configuration import *
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, log_loss, accuracy_score


def load_input(inPath):
    input_df = pd.read_csv(inPath, sep=';')
    files = pd.Series(input_df['uri'].apply(lambda x: ast.literal_eval(x)))[0]
    train = pd.read_csv(files['train_file_path'], index_col=[0])
    predict = pd.read_csv(files['prediction_input'], index_col=[0])
    return files, train, predict


def perform_undersampling(df):
    # optional
    # random.seed(42)

    # assuming last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # assuming train data is imbalanced
    y_1 = y[y.values == 1]
    y_0 = y[y.values == 0]
    X_1 = X.loc[y_1.index]
    X_0 = X.loc[y_0.index]
    print(f"Share of 1 in the train set: { '{:.2%}'.format(y_1.shape[0]/y.shape[0])}")

    split = pd.Series(data=np.random.randint(0, 1 / SHARE_OF_TT_SPLIT, X_1.shape[0]), index=X_1.index)

    X_1_test = X_1.loc[split[split.values == 0].index]
    X_1_train = X_1.loc[split[split.values > 0].index]

    X_train = pd.concat([X_1_train, X_0.sample(X_1_train.shape[0])])
    y_train = y.loc[X_train.index]

    X_test = pd.concat([X_1_test, X_0.sample(X_1_test.shape[0])])
    y_test = y.loc[X_test.index]

    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    trained_models = {}

    # collecting input from weak classifiers
    predictions_train = pd.DataFrame()
    predictions_train['y_test'] = y_test

    for n, m in MODELS:
        m.fit(X_train, y_train)
        y_pred = pd.Series(data=m.predict(X_test), index=X_test.index, name=f'{n}')
        predictions_train = predictions_train.merge(y_pred, how='outer', left_index=True, right_index=True)

        # keep fitted models as variable + dump to file
        trained_models[f'{n}'] = m
        # optional
        # pickle.dump(m, open(f'weak_model_{n}.sav', 'wb'))



    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(predictions_train.iloc[:, 1:],
                                                                predictions_train['y_test'],
                                                                test_size=0.3)
    voting_model = VOTING[0][1]
    voting_model.fit(X_train_p, y_train_p)
    y_pred_p = voting_model.predict(X_test_p)
    # optional
    # pickle.dump(m, open(f'voting_model_{VOTING[0][0]}.sav', 'wb'))

    tn, fp, fn, tp = confusion_matrix(y_test_p, y_pred_p).ravel()
    results = {
        'voting model': voting_model,
        'Accuracy': '{:.4f}'.format(accuracy_score(y_test_p, y_pred_p)),
        'F1': '{:.4f}'.format(f1_score(y_test_p, y_pred_p)),
        'share of incorrect predictions': '{:.2%}'.format((fp + fn)/(tn + fp + fn + tp)),
        'number of incorrect predictions': fp+fn
    }
    print(f'Results on training under sampled data\n{results}')
    return trained_models, voting_model


def make_prediction(predict, outPath, trained_models, voting_model):
    predictions = pd.DataFrame()
    for n in trained_models.keys():
        y_pred = pd.Series(data=trained_models[n].predict(predict), index=predict.index, name=f'{n}')
        predictions = predictions.merge(y_pred, how='outer', left_index=True, right_index=True)

    prediction_results = voting_model.predict(predictions)
    results_df = pd.DataFrame(data=prediction_results, index=predict.index, columns=['target'])
    results_df.to_csv(outPath, index=True)
    n_1 = results_df[results_df['target']==1].shape[0]
    n_1_pct = n_1/results_df.shape[0]
    print(f"Detected {n_1} customers ({'{:.2%}'.format(n_1_pct)}) who are going to convert")
    print("Results of prediction saved to results.csv")

