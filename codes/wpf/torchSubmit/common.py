import numpy as np
import pandas as pd
import torch
from torch import nn


def data_smooth(data, steps=2):
    if steps > 1:
        x_enc = data[1]  # (batch, seq, feature)
        total_len = x_enc.size(1)
        smooth_len = total_len - steps + 1
        x_enc_sum = 0.
        for i in range(steps):
            x_enc_sum += x_enc[:, i:i + smooth_len, :]
        x_enc = x_enc_sum / steps
        data[1] = x_enc
    return data


def overall_score(diff_arr):
    diff_arr = diff_arr[diff_arr == diff_arr]
    mae = diff_arr.abs().mean()
    rmse = np.sqrt((diff_arr ** 2).mean())
    return (mae + rmse) / 2.0


def cal_score(turb_period, y_true, y_pred):
    """
    :param turb_period: (N, 2)
    :param y_true: (N, 288)
    :param y_pred: (N, 288)
    :return: score = (RMSE + MAE) / 2, (N, 1)
    """
    n_times = y_pred.shape[1]
    true = np.concatenate((turb_period, y_true), axis=1)
    df_true = pd.DataFrame(true, columns=["turb", "time"] + list(range(1, n_times + 1)))

    pred = np.concatenate((turb_period, y_pred), axis=1)
    df_pred = pd.DataFrame(pred, columns=["turb", "time"] + list(range(1, n_times + 1)))

    df_diff = df_true.set_index(['turb', 'time']) - df_pred.set_index(['turb', 'time'])
    turb_score = df_diff.apply(overall_score, axis=1).to_frame("score").reset_index()
    farm_score = turb_score.groupby("time")['score'].sum()

    return farm_score, df_diff


def evaluate_loader(model, dataloader, device, dp, smooth_steps=1):
    """
    evaluate model on val or test dataset
    :param model: trained model
    :param dataloader: val dataloader or test dataloader
    :param device: torch.device()
    :param dp: data_prepare
    :return: overall score
    """
    model = model.to(device)
    turb_periods = []
    y_trues = []
    y_preds = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data_smooth(data, smooth_steps)
            data = [d.to(device) for d in data]
            turb_periods.append(data[0].cpu().detach())
            y_pred = model(data)
            y_pred = y_pred[:, :, -1]  # (bs, output_len, out_var) => (bs, output_len)
            y_preds.append(y_pred.cpu().detach())
            y_true = data[3].cpu().detach()[:, :, -1]
            y_trues.append(y_true)  # (bs, output_len, out_var) => (bs, output_len)
    model.train()
    turb_period = torch.cat(turb_periods, dim=0)
    y_true = torch.cat(y_trues, dim=0)
    y_pred = torch.cat(y_preds, dim=0)

    # inverse transform, and turn tensor to np.array
    y_pred = dp.inverse_transform_Patv(y_pred)
    y_true = dp.inverse_transform_Patv(y_true)
    score, df_diff = cal_score(turb_period, y_true=y_true, y_pred=y_pred)
    return score, df_diff


def evaluate_loader_models(models, dataloader, device, dp, smooth_steps=1):
    """
    evaluate model on val or test dataset
    :param models: trained models
    :param dataloader: val dataloader or test dataloader
    :param device: torch.device()
    :param dp: data_prepare
    :return: overall score
    """
    y_pred_list = []
    for model in models:
        model = model.to(device)
        y_preds = []
        model.eval()
        with torch.no_grad():
            for data in dataloader:
                data = data_smooth(data, smooth_steps)
                data = [d.to(device) for d in data]
                y_pred = model(data)
                y_pred = y_pred[:, :, -1]  # (bs, output_len, out_var) => (bs, output_len)
                y_preds.append(y_pred.cpu().detach())
        model.train()
        y_pred = torch.cat(y_preds, dim=0)
        y_pred_list.append(y_pred)

    turb_periods = [data[0].cpu().detach() for data in dataloader]
    turb_period = torch.cat(turb_periods, dim=0)
    y_trues = [data[3].cpu().detach()[:, :, -1] for data in dataloader]
    y_true = torch.cat(y_trues, dim=0)
    y_pred = sum(y_pred_list) / len(y_pred_list)
    # inverse transform, and turn tensor to np.array
    y_pred = dp.inverse_transform_Patv(y_pred)
    y_true = dp.inverse_transform_Patv(y_true)
    score, _ = cal_score(turb_period, y_true=y_true, y_pred=y_pred)
    return score.mean() / 1000.0


class WPFLoss(nn.Module):
    """
    - loss = 1/2 * (RMSE + MAE)
    - ignore Missing values and Unknown values
    """

    def __init__(self, alpha=0.0001):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.alpha = alpha

    def forward(self, y_true, y_pred, model=None):
        """
        :param y_true: (batch, 288, 1)
        :param y_pred: (batch, 288, 1)
        """
        assert y_true.shape == y_pred.shape, "Wrong dimension with y_true and y_pred!"
        y_pred[y_true != y_true] = 0  # fill na with 0
        y_true[y_true != y_true] = 0  # fill na with 0
        mae = self.mae(y_true, y_pred)
        rmse = torch.sqrt(self.mse(y_true, y_pred))
        regularizer = 0.
        if model:
            regularizer = sum([torch.abs(para).sum() for para in model.parameters()])

        return (mae + rmse) / 2.0 + self.alpha * regularizer
