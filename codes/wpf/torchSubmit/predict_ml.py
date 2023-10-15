import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


def make_dataset(wpf_file):
    df = pd.read_csv(wpf_file)
    df['Hour'] = pd.to_datetime(df['Tmstamp']).dt.hour

    df.loc[df["Patv"] < 0, 'Patv'] = np.nan
    # unknown values
    df.loc[(df['Patv'] == 0) & (df['Wspd'] > 2.5), 'Patv'] = np.nan
    df.loc[(df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89), 'Patv'] = np.nan
    # abnormal values
    df.loc[(df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) | (df['Ndir'] > 720), 'Patv'] = np.nan

    tmp = df.groupby(['Day', 'Hour'])['Patv'].mean().reset_index()
    feat_len = 72
    pred_len = 48

    X_list = []
    y_list = []
    for j in range(len(tmp) - feat_len - pred_len):
        x = tmp.iloc[j:j + feat_len, 2].values
        y = tmp.iloc[j + feat_len:j + feat_len + pred_len, 2].values
        # filter samples with nan
        if np.isnan(x).any() or np.isnan(y).any() > 0:
            continue
        X_list.append(x)
        y_list.append(y)
    X = np.array(X_list)
    y = np.array(y_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def rf_train48(wpf_file, settings, model_name="rf_mean48.pkl"):
    X_train, X_test, y_train, y_test = make_dataset(wpf_file)
    rf = RandomForestRegressor(max_depth=5).fit(X_train, y_train)
    model_path = os.path.join(settings['checkpoints'], "models", model_name)
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"RF trained finished, and saved with path: {model_path}")


def lgb_train48(wpf_file, settings, model_name="lgb_mean48.pkl", **kwargs):
    X_train, X_test, y_train, y_test = make_dataset(wpf_file)
    n_outputs = y_train.shape[1]
    lgb_models = {}
    for i in range(n_outputs):
        lgb_models[i] = LGBMRegressor(**kwargs).fit(X_train, y_train[:, i])
    model_path = os.path.join(settings['checkpoints'], "models", model_name)
    with open(model_path, "wb") as f:
        pickle.dump(lgb_models, f)
    print(f"LGB trained finished, and saved with path: {model_path}")


def load_model(settings, model_name):
    path = os.path.join(settings['checkpoints'], "models", model_name)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def make_X_test(wpf_file):
    df = pd.read_csv(wpf_file)
    df.loc[(df['Patv'] == 0) & (df['Wspd'] > 2.5), 'Patv'] = np.nan
    df.loc[(df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89), 'Patv'] = np.nan
    # abnormal values
    df.loc[(df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) | (df['Ndir'] > 720), 'Patv'] = np.nan

    df['DT'] = df['Day'].astype(str).str.zfill(3) + " " + df['Tmstamp']
    patv = df.pivot(index="DT", columns="TurbID", values="Patv").fillna(method="pad").fillna(method="bfill").T

    return patv


def rf_predict48(wpf_file, settings):
    rf = load_model(settings, "rf_mean48.pkl")
    patv = make_X_test(wpf_file)
    X = patv.iloc[:, -432:]
    X = X.values.reshape((len(X), 72, 6)).mean(axis=2)
    y_pred = rf.predict(X)
    y_pred = y_pred.repeat(6, axis=1)
    y_pred = np.expand_dims(y_pred, axis=2)
    return y_pred


def lgb_predict48(wpf_file, settings):
    models = load_model(settings, "lgb_mean48.pkl")
    n_outputs = len(models)
    patv = make_X_test(wpf_file)
    X = patv.iloc[:, -432:]
    X = X.values.reshape((len(X), 72, 6)).mean(axis=2)
    y_pred = np.empty((len(X), n_outputs))
    for i in range(n_outputs):
        y_pred[:, i] = models[i].predict(X)
    y_pred = y_pred.repeat(6, axis=1)
    y_pred = np.expand_dims(y_pred, axis=2)
    return y_pred


if __name__ == "__main__":
    from prepare import prep_env

    settings = prep_env()
    wpf_file = os.path.join(settings["data_path"], settings["filename"])
    rf_train48(wpf_file, settings)
    kwargs = {"max_depth": 10}
    lgb_train48(wpf_file, settings, **kwargs)
