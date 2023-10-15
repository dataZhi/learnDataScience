import os
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader


class DataPrepare:
    def __init__(self, wpf_file, turb_file, checkpoint_dir, interpolate="linear", feature_derive=False, n_neighbors=8,
                 n_turb=134, flag="train", out_var=1, dates=None):
        self.interpolate = interpolate
        self.feature_derive = feature_derive
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.n_neighbors = n_neighbors
        self.n_turb = n_turb
        self.flag = flag
        self.out_var = out_var
        self.dates = dates

        self.feature_cols = ['Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Wspd', 'Wdir', 'Etmp', 'Patv']
        self.decoder_cols = ['Patv'] if out_var == 1 else ['Wspd', 'Patv']
        df = pd.read_csv(wpf_file)
        if dates is not None:
            start, end = dates
            df = df.query(f"Day>={start} and Day<={end}")
        self.df = self._process_wpf_data(df)

        self.all_cols = ['Hour'] + self.feature_cols + self.label_cols
        self._transform_features()

        self.n_period = len(df) // self.n_turb
        self.data = np.empty((self.n_turb, self.n_period, len(self.all_cols)))
        for i in range(self.n_turb):
            self.data[i] = self.df.query(f"TurbID=={i + 1}")[self.all_cols].values
        self._transform_hour()

        self.location = pd.read_csv(turb_file).iloc[:n_turb]
        if n_neighbors > 0:
            self._merge_neighbors(n_neighbors)

        data = self.data.transpose(0, 2, 1).reshape((self.n_turb, len(self.all_cols), -1, 6))
        data = data.mean(axis=3)
        self.data_agg = data.transpose(0, 2, 1)

    @staticmethod
    def _fileter_outlier(df, col, factor=1.5):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        return (df[col] > (Q3 + factor * IQR)) | (df[col] < (Q1 - factor * IQR))

    def _process_wpf_data(self, df):
        """
        process Missing values and Unknown values, interpolate
        :param df: raw df
        :return: processed df
        """
        # missing values
        df.loc[df["Patv"] < 0, 'Patv'] = np.nan

        # unknown values
        df.loc[(df['Patv'] == 0) & (df['Wspd'] > 2.5), 'Patv'] = np.nan
        df.loc[(df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89), 'Patv'] = np.nan

        # abnormal values
        df.loc[(df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) | (df['Ndir'] > 720), 'Patv'] = np.nan

        # backup Patv and Wspd columns
        self.label_cols = [col + "Raw" for col in self.decoder_cols]
        label_raw = df[self.decoder_cols].copy()

        # abnormal values in other column
        df.loc[(df['Wdir'] < -180) | (df['Wdir'] > 180), 'Wdir'] = np.nan
        df.loc[(df['Ndir'] < -720) | (df['Ndir'] > 720), 'Ndir'] = np.nan
        df.loc[df['Pab1'] > 89, 'Pab1'] = np.nan
        df.loc[df['Pab2'] > 89, 'Pab2'] = np.nan
        df.loc[df['Pab3'] > 89, 'Pab3'] = np.nan
        df.loc[self._fileter_outlier(df, 'Etmp'), 'Etmp'] = np.nan
        df.loc[self._fileter_outlier(df, 'Itmp'), 'Itmp'] = np.nan

        # interpolate
        if self.interpolate is not None:
            df = df.groupby("TurbID").apply(lambda sdf: sdf.interpolate(method=self.interpolate))

        # feature derive
        if self.feature_derive:
            df['Ddir'] = df['Wdir'] - df['Ndir']
            df['Dtmp'] = df['Etmp'] - df['Itmp']
            df['Wspd3'] = df['Wspd'] ** 3
            self.feature_cols.extend(['Ddir', 'Dtmp', 'Wspd3'])

        df[self.label_cols] = label_raw
        # extract hour column
        df['Hour'] = pd.to_datetime(df['Tmstamp']).dt.hour
        return df

    def _transform_features(self):
        scalar_file = os.path.join(self.checkpoint_dir, "scaler.pkl")
        if self.flag == "train":
            self.scalars = {}
            for col in self.all_cols:
                if col == "Hour":
                    continue
                X_tmp = self.df[[col]].values
                self.scalars[col] = MinMaxScaler().fit(X_tmp)
            with open(scalar_file, 'wb') as f:
                pickle.dump(self.scalars, f)
        else:
            with open(scalar_file, 'rb') as f:
                self.scalars = pickle.load(f)
        for col in self.all_cols:
            if col.startswith("Hour"):
                continue
            self.df[[col]] = self.scalars[col].transform(self.df[[col]])

    def inverse_transform_Patv(self, y):
        """
        :param y: wind power of forecast, (batch, 288)
        :return: inverse_transformed wind power, (batch, 288)
        """
        y = y.detach().cpu().numpy()
        y_reverse = self.scalars["Patv"].inverse_transform(y.reshape(-1, 1))
        y_reverse = y_reverse.reshape(*y.shape)
        return y_reverse

    def _transform_hour(self):
        hour_idx = self.all_cols.index("Hour")
        self.data[:, :, hour_idx] = self.data[:, :, hour_idx] / 23

    def _get_neighbors(self, n_neighbors):
        self.location_scalar = MinMaxScaler()
        X = self.location_scalar.fit_transform(self.location[['x', 'y']])
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(X)
        distance, ids = neigh.kneighbors(X)
        self.neighbor_distance = distance
        self.neighbor_ids = ids

    def _merge_neighbors(self, n_neighbors):
        self._get_neighbors(n_neighbors)
        # extract features from neighbor turbs
        neighbor_distance = self.neighbor_distance[:, 1:]  # shape: (n_turb, 4)
        neighbor_distance = np.expand_dims(neighbor_distance, 1).repeat(self.n_period, axis=1)  # (n_turb, n_period, 4)
        self.data = np.concatenate((self.data, neighbor_distance), axis=2)
        self.all_cols.extend([f"Distance_{i}" for i in range(1, n_neighbors)])

        feature_cols = ['Wspd', 'Wdir', 'Etmp', 'Patv']
        feature_idx = [idx for idx, col in enumerate(self.all_cols) if col in feature_cols]
        for i in range(1, n_neighbors):
            neighbor_ids = self.neighbor_ids[:, i]
            neighbor_features = self.data[neighbor_ids][:, :, feature_idx]
            self.data = np.concatenate((self.data, neighbor_features), axis=2)
            self.all_cols.extend([col + "_" + str(i) for col in feature_cols])

    def make_dataloader(self, input_len=144 * 3, output_len=288, batch_size=512,
                        k_fold=0, n_models=1, turbID=None, data_agg=False):
        st_idx = [
                     idx for idx, col in enumerate(self.all_cols) if col.startswith("Hour")
                 ] + [
                     idx for idx, col in enumerate(self.all_cols) if col.startswith("Distance")
                 ]  # index of spatio and temporal features with specified order

        enc_idx = st_idx + [
            self.all_cols.index(col) for col in self.feature_cols
        ] + [
                      self.all_cols.index(col) for i in range(1, self.n_neighbors) for col in
                      [f"Wspd_{i}", f"Wdir_{i}", f"Etmp_{i}", f"Patv_{i}"]
                  ]  # index of st features and own/neighbor features with specified order

        dec_idx = [self.all_cols.index(col) for col in self.decoder_cols]  # index of [Wspd, Patv]

        label_idx = [self.all_cols.index(col) for col in self.label_cols]  # index of [WspdRaw, PatvRaw]

        if data_agg:
            data = self.data_agg
            dataset = WPFDatasetAgg(data, st_idx, enc_idx, dec_idx, label_idx,
                                    input_len=input_len, output_len=output_len, flag=self.flag,
                                    k_fold=k_fold, n_models=n_models, turbID=turbID)
        else:
            data = self.data
            dataset = WPFDataset(data, st_idx, enc_idx, dec_idx, label_idx,
                                 input_len=input_len, output_len=output_len, flag=self.flag,
                                 k_fold=k_fold, n_models=n_models, turbID=turbID)
        batch_size = batch_size if self.flag in ("train", "val") else 64
        shuffle = True if self.flag in ("train", "val") else False
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader


class WPFDataset(Dataset):
    def __init__(self, data, st_idx, enc_idx, dec_idx, label_idx,
                 input_len=144 * 5, output_len=288, flag="train", k_fold=0, n_models=1,
                 turbID=None):
        self.data = torch.tensor(data, dtype=torch.float)  # data: (n_turb, n_period, n_feature)
        if flag == "test":  # only load the last sample
            self.data = self.data[:, -input_len:, :]
        self.n_turb = 134 if turbID is None else 1
        self.turbID = turbID
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len
        self.flag = flag
        self.k_fold = k_fold
        self.n_models = n_models

        self.st_idx = st_idx
        self.enc_idx = enc_idx
        self.dec_idx = dec_idx
        self.label_idx = label_idx

        assert input_len >= 144, "input_len < 144, and will introduce error when generating x_dec"

    def __getitem__(self, index):
        """
        :param index: sample index
        :return: [[turbID, cur_period], x_enc, x_dec, y]
        """
        turbID = index % self.n_turb if self.turbID is None else self.turbID - 1
        period = index // self.n_turb
        period = period * self.n_models + self.k_fold  # sample data for multi models
        cur_period = period + self.input_len
        data = self.data[turbID, period:period + self.total_len, :]

        turb_period = torch.tensor([turbID, cur_period], dtype=torch.long)

        x_enc = data[:self.input_len, self.enc_idx]  # (input_len, input_size)

        x_dec = data[self.input_len - 144:self.input_len, self.st_idx].repeat(2, 1)  # (288, st_idx)
        x_dec_0 = data[self.input_len - 1:self.input_len, self.st_idx]  # (1, st_idx)  initial status of decoder
        x_dec = torch.cat((x_dec_0, x_dec), dim=0)  # (289, st_idx)

        label_dummy = data[self.input_len - 1, self.dec_idx].repeat(289, 1)
        x_dec = torch.cat((x_dec, label_dummy), dim=1)  # (289, st_idx+dec_idx)

        if self.flag in ("train", "val"):
            y = data[self.input_len:self.total_len, self.label_idx]
            return turb_period, x_enc, x_dec, y
        else:
            return turb_period, x_enc, x_dec

    def __len__(self):
        if self.flag in ("train", "val"):
            total_len = self.n_turb * (self.data.size(1) - self.total_len + 1)
            return total_len // self.n_models
        else:
            return self.n_turb


class WPFDatasetAgg(Dataset):
    """
    make dataset with a larger temporal scale, i.e. an hour for one time slot instead of 10 minutes
    """
    def __init__(self, data, st_idx, enc_idx, dec_idx, label_idx,
                 input_len=72, output_len=48, flag="train", k_fold=0, n_models=1,
                 turbID=None):
        self.data = torch.tensor(data, dtype=torch.float)  # data: (n_turb, n_period, n_feature)
        if flag == "test":
            self.data = self.data[:, -input_len:, :]
        self.n_turb = 134 if turbID is None else 1
        self.turbID = turbID
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len
        self.flag = flag
        self.k_fold = k_fold
        self.n_models = n_models

        self.st_idx = st_idx
        self.enc_idx = enc_idx
        self.dec_idx = dec_idx
        self.label_idx = label_idx

        assert input_len >= 24, "input_len < 24, and will introduce error when generating x_dec"

    def __getitem__(self, index):
        """
        :param index: sample index
        :return: [[turbID, cur_period], x_enc, x_dec, y]
        """
        turbID = index % self.n_turb if self.turbID is None else self.turbID - 1
        period = index // self.n_turb
        period = period * self.n_models + self.k_fold  # sample data for multi models
        cur_period = period + self.input_len
        data = self.data[turbID, period:period + self.total_len, :]

        turb_period = torch.tensor([turbID, cur_period], dtype=torch.long)

        x_enc = data[:self.input_len, self.enc_idx]  # (input_len, input_size)

        x_dec = data[self.input_len - 24:self.input_len, self.st_idx].repeat(2, 1)  # (48, st_idx)
        x_dec_0 = data[self.input_len - 1:self.input_len, self.st_idx]  # (1, st_idx)
        x_dec = torch.cat((x_dec_0, x_dec), dim=0)  # (49, st_idx)

        label_dummy = data[self.input_len - 1, self.dec_idx].repeat(49, 1)
        x_dec = torch.cat((x_dec, label_dummy), dim=1)  # (49, st_idx+dec_idx)

        if self.flag in ("train", "val"):
            y = data[self.input_len:self.total_len, self.label_idx]
            return turb_period, x_enc, x_dec, y
        else:
            return turb_period, x_enc, x_dec

    def __len__(self):
        if self.flag in ("train", "val"):
            return self.n_turb * ((self.data.size(1) - self.total_len + 1) // self.n_models)
        else:
            return self.n_turb
