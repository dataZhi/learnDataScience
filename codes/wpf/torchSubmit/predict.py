import os
import torch
from dataset import DataPrepare
from predict_ml import rf_predict48, lgb_predict48
import warnings

warnings.filterwarnings("ignore")


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions
    """
    checkpoint_dir = settings['checkpoints']
    wpf_file = settings["path_to_test_x"]
    turb_file = os.path.join(checkpoint_dir, "data", settings['turb_file'])
    n_neighbors = settings['n_neighbors']
    input_len = settings['input_len']
    output_len = settings['output_len']
    out_var = settings['out_var']
    model_prefix = settings['model_prefix']
    device = settings['device']
    trend_with_ml = settings['trend_with_ml']
    trend4calibrate = settings['trend4calibrate']

    model_path = os.path.join(checkpoint_dir, "models")
    files = os.listdir(model_path)
    model_files = [file for file in files if file.startswith(model_prefix) and not file.__contains__("dataAgg")]
    model_files_agg = [file for file in files if file.startswith(model_prefix) and file.__contains__("dataAgg")]
    assert len(model_files) > 0, "no models found"

    edp = DataPrepare(wpf_file, turb_file, checkpoint_dir, flag="test",
                      out_var=out_var, n_neighbors=n_neighbors, dates=None)
    testloader = edp.make_dataloader(input_len=input_len, output_len=output_len)
    testloader_agg = edp.make_dataloader(input_len=72, output_len=48, data_agg=True)
    models = []
    for model_file in model_files:
        models.append(torch.load(os.path.join(model_path, model_file), map_location="cpu"))
    models_agg = []
    for model_file in model_files_agg:
        models_agg.append(torch.load(os.path.join(model_path, model_file), map_location="cpu"))

    results = []
    with torch.no_grad():
        for model in models:
            model.eval()
            model = model.to(device)
            predictions = []
            for data in testloader:
                data = [d.to(device) for d in data]
                prediction = model(data)  # (1, 288, out_var)
                prediction = prediction[:, :, -1:]  # (1, 288, 1)
                predictions.append(prediction)
            result = torch.cat(predictions, dim=0)  # (134, 288, 1)
            results.append(result)
    results = sum(results) / len(results)
    results = edp.inverse_transform_Patv(results)

    if models_agg.__len__() > 0 and not trend_with_ml:
        results_agg = []
        with torch.no_grad():
            for model in models_agg:
                model.eval()
                model = model.to(device)
                predictions = []
                for data in testloader_agg:
                    data = [d.to(device) for d in data]
                    prediction = model(data)  # (1, 288, out_var)
                    prediction = prediction[:, :, -1:]  # (1, 288, 1)
                    predictions.append(prediction)
                result = torch.cat(predictions, dim=0)  # (134, 288, 1)
                results_agg.append(result)
        results_agg = sum(results_agg) / len(results_agg)
        results_agg = edp.inverse_transform_Patv(results_agg)
        trend = results_agg.repeat(6, axis=1)
    elif trend_with_ml:
        rf_trend = rf_predict48(wpf_file, settings)
        lgb_trend = lgb_predict48(wpf_file, settings)
        coef = 0.5
        trend = coef * lgb_trend + (1 - coef) * rf_trend
    else:
        trend = results

    # fusion predictions with trend results
    if trend4calibrate:
        trend_mean = trend.reshape((len(trend), -1, 6)).mean(axis=2, keepdims=True)
        results_mean = results.reshape((len(results), -1, 6)).mean(axis=2, keepdims=True)
        delta_trend = results_mean - trend_mean
        results = results - delta_trend.repeat(6, axis=1)
    else:
        pre_k = 36
        results[:, :pre_k, :] = (results[:, :pre_k, :] + trend[:, :pre_k, :]) / 2.0

    return results
