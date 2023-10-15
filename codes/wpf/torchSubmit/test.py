import os
import torch
from dataset import DataPrepare
from common import evaluate_loader_models
from predict_ml import rf_predict48, lgb_predict48
import warnings

warnings.filterwarnings("ignore")


def test_metrics(settings):
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
    wpf_file = os.path.join(settings["data_path"], settings["filename"])
    turb_file = os.path.join(checkpoint_dir, "data", settings['turb_file'])
    n_neighbors = settings['n_neighbors']
    input_len = settings['input_len']
    output_len = settings['output_len']
    out_var = settings['out_var']
    model_prefix = settings['model_prefix']
    device = settings['device']

    model_path = os.path.join(checkpoint_dir, "models")
    files = os.listdir(model_path)
    model_files = [file for file in files if file.startswith(model_prefix) and not file.__contains__("dataAgg")]
    assert len(model_files) > 0, "no models found"

    vdp = DataPrepare(wpf_file, turb_file, checkpoint_dir, flag="val",
                      out_var=out_var, n_neighbors=n_neighbors, dates=(170, 184))
    valloader = vdp.make_dataloader(input_len=input_len, output_len=output_len)
    models = []
    for model_file in model_files:
        models.append(torch.load(os.path.join(model_path, model_file), map_location="cpu"))

    score = evaluate_loader_models(models, valloader, device, vdp)

    return score


if __name__ == "__main__":
    from prepare import prep_env
    settings = prep_env()
    score = test_metrics(settings)
    print(score)
