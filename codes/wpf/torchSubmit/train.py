import os
import time
from copy import deepcopy
from multiprocessing import Pool
import torch
from torch import optim
from dataset import DataPrepare
from stan import STAN
from common import evaluate_loader, WPFLoss
import warnings

warnings.filterwarnings("ignore")


def train_one(args):
    checkpoint_dir = args['checkpoints']
    wpf_file = os.path.join(args["data_path"], args["filename"])
    turb_file = os.path.join(checkpoint_dir, "data", args['turb_file'])
    out_var = args['out_var']
    n_neighbors = args['n_neighbors']
    tdp = DataPrepare(wpf_file, turb_file, checkpoint_dir, flag="train",
                      out_var=out_var, n_neighbors=n_neighbors, dates=(70, 160))
    vdp = DataPrepare(wpf_file, turb_file, checkpoint_dir, flag="val",
                      out_var=out_var, n_neighbors=n_neighbors, dates=(161, 184))

    data_agg = args['data_agg']
    input_len = 72 if data_agg else args['input_len']
    output_len = 48 if data_agg else args['output_len']
    out_var = args['out_var']
    batch_size = args['batch_size']

    k_fold = args['k_fold']
    n_models = args['n_models']

    trainloader = tdp.make_dataloader(input_len=input_len, output_len=output_len, batch_size=batch_size,
                                      k_fold=k_fold, n_models=n_models, data_agg=data_agg)
    valloader = vdp.make_dataloader(input_len=input_len, output_len=output_len, batch_size=batch_size,
                                    k_fold=k_fold, n_models=n_models, data_agg=data_agg)

    device = args['device']
    n_neighbors = args['n_neighbors']
    num_heads = args['num_heads']

    data = trainloader.__iter__().next()
    enc_size = data[1].size(2)
    dec_size = data[2].size(2)
    print("=" * 80,
          f"\nDataset prepared, enc_size: {enc_size}, dec_size: {dec_size}, loader size: {trainloader.__len__()}")

    hidden_as_input = args['hidden_as_input']
    hidden_as_output = args['hidden_as_output']
    model = STAN(enc_size=enc_size, dec_size=dec_size, output_len=output_len, out_var=out_var,
                 n_neighbors=n_neighbors, num_heads=num_heads,
                 hidden_as_input=hidden_as_input, hidden_as_output=hidden_as_output).to(device)
    num_para = sum([para.nelement() for para in model.parameters()])
    print("=" * 80, f"\nModel initialized, and total number of parameters is: {num_para}")

    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = WPFLoss()

    dataAgg = "_dataAgg" if data_agg else ""
    model_name = args['model_prefix'] + dataAgg + f"_{k_fold}.pkl"
    model_file = os.path.join(checkpoint_dir, "models", model_name)
    print(f"================= Model training starts, and use device {device} ===================")
    time.sleep(1)

    patience = 10
    best_score = 999
    bad_count = 0
    for epoch in range(10):
        loss_total = 0.
        model.train()
        for idx, data in enumerate(trainloader):
            """
            data: [[turbID, cur_period], x_enc, x_dec, y]
            x_enc: (batch, seq_len, enc_size), default enc_size=54
            x_dec: (batch, seq_len, dec_size), default dec_size=29
            y: (batch, output_len, out_var)
            """
            data = [d.to(device) for d in data]
            y_pred = model(data)
            y_true = data[3]
            loss = criterion(y_true, y_pred, model=model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

            if idx % 100 == 0:
                val_score, _ = evaluate_loader(model, valloader, device, vdp)
                cur_score = val_score.mean() / 1000
                print(epoch, idx, loss_total, cur_score)
                loss_total = 0.
                if cur_score < best_score:
                    best_score = cur_score
                    torch.save(model, model_file)
                else:
                    bad_count += 1
            if bad_count >= patience:
                print(f"Model training stopped, with {bad_count} times, and best_socre: {best_score}.")
                break
        if bad_count >= patience:
            print(f"Finally: model training stopped, with {bad_count} times, and best_socre: {best_score}.")
            break


def train_multi(settings):
    checkpoint_dir = settings['checkpoints']
    wpf_file = os.path.join(settings["data_path"], settings["filename"])
    turb_file = os.path.join(checkpoint_dir, "data", settings['turb_file'])
    out_var = settings['out_var']
    n_neighbors = settings['n_neighbors']
    num_workers = settings['num_workers']
    tdp = DataPrepare(wpf_file, turb_file, checkpoint_dir, flag="train",
                      out_var=out_var, n_neighbors=n_neighbors, dates=(70, 160))
    vdp = DataPrepare(wpf_file, turb_file, checkpoint_dir, flag="val",
                      out_var=out_var, n_neighbors=n_neighbors, dates=(161, 184))

    n_models = settings['n_models']
    args_list = [deepcopy(settings) for _ in range(n_models)]
    for i, args in enumerate(args_list):
        args['k_fold'] = i
        args['tdp'] = tdp
        args['vdp'] = vdp

    # TODO: multiprocessing.Pool
    for args in args_list:
        train_one(args)
    pool = Pool(num_workers)
    pool.map(train_one, args_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    from prepare import prep_env

    args = prep_env()
    train_one(args)
