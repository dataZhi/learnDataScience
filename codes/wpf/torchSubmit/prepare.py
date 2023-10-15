# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
"""
import torch


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "path_to_test_x": "./test_x",
        "path_to_test_y": "./test_y",
        "data_path": "../data",
        "filename": "wpf_245days.csv",
        "turb_file": "turb_location.csv",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "checkpoints",
        "model_prefix": "STAN",
        "data_agg": False,
        "trend_with_ml": True,
        "trend4calibrate": False,
        "n_models": 4,
        "k_fold": 0,
        "n_neighbors": 8,
        "num_heads": 8,
        "input_len": 144*5,
        "output_len": 288,
        "hidden_as_input": False,
        "hidden_as_output": False,
        "start_col": 3,
        "in_var": 10,
        "out_var": 1,
        "day_len": 144,
        "train_size": 214,
        "val_size": 31,
        "total_size": 245,
        "lstm_layer": 2,
        "dropout": 0.05,
        "num_workers": 5,
        "train_epochs": 10,
        "batch_size": 64,
        "patience": 3,
        "lr": 1e-4,
        "lr_adjust": "type1",
        "gpu": 0,
        "capacity": 134,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "pytorch",
        "is_debug": True,
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    }

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
