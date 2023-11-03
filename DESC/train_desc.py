import sys, os, csv
import json
import argparse
import time
import random
random.seed(2022)
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from transformers import AutoImageProcessor, SwinForMaskedImageModeling
except: pass

from dataset_desc import CalXXXDataset
from model_desc import CalXXXModel
from utils import AverageMeter, meter_to_str, get_opt, update_opt, compute_pairwise_loss



RBIT = 4

def main(param):
    np.random.seed(param.seed)
    torch.manual_seed(param.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(param.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if param.cpu:
        device = torch.device("cpu")

    if torch.cuda.device_count() > 1:
        param.batch_size *= torch.cuda.device_count()

    if not os.path.exists(param.model_folder):
        os.mkdir(param.model_folder)

    need_keys = param.need_keys.split(" ")
    train_dataset = CalXXXDataset(sample_path=param.sample_path, split_path=param.split_path, 
                                  cross_path=param.cross_path, need_keys=need_keys, 
                                  label_name=param.label_name, pctr_label_name=param.pctr_label_name, debug=param.debug)
    print("#training samples", train_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True, num_workers=param.workers)

    if os.path.exists(param.test_sample_path):
        test_dataset = CalXXXDataset(sample_path=param.test_sample_path, split_path=param.split_path, 
                                     cross_path=param.cross_path, cross_value=train_dataset.crossfield2value,
                                     need_keys=need_keys, label_name=param.label_name, pctr_label_name=param.pctr_label_name,
                                     debug=param.debug, train_flag=False)
        test_dataloader = DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False, num_workers=param.workers)
    else:
        test_dataset, test_dataloader = None, None

    max_value = {key: 0 for key in train_dataset.need_keys}
    for key in train_dataset.max_value:
        max_value[key] = max(max_value[key], train_dataset.max_value[key])
    for key in test_dataset.max_value:
        max_value[key] = max(max_value[key], test_dataset.max_value[key])

    print("need_keys", train_dataset.need_keys)
    print("max_value", max_value)
    fc_hidden_size = [int(v.strip()) for v in param.fc_hidden_size_str.split(',')]
    if len(fc_hidden_size) != 4:
        print("fc size not equal 4", len(fc_hidden_size))
        sys.exit(-1)
    model = CalXXXModel(emb_size=param.emb_size, bin_name2bin_num=max_value,
                     field_name_list=train_dataset.need_keys, fc_hidden_size=fc_hidden_size,
                     drop_prob=param.dropout, stop_gradient=param.stop_gradient, ts_weight_folder=param.ts_weight_folder)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.0)
    model.apply(init_weights)
    #print("model", model)

    if os.path.exists(param.checkpoint_path):
        print("load state", param.checkpoint_path)
        model.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
        print("load complete")
    print("model", model)
    if torch.cuda.device_count() > 1:
        print("#gpus", torch.cuda.device_count())
        model = nn.DataParallel(model)
    train(model, device, train_dataset, train_dataloader, test_dataset, test_dataloader, param)

def train(model, device, train_dataset, train_dataloader, test_dataset, test_dataloader, param, start_epoch=0):
    model.train()
    model = model.to(device)
    # model.ts_weight = model.ts_weight.to(device)

    criterion = F.binary_cross_entropy  # binary_cross_entropy  # Cross loss
    eval_step = int(len(train_dataloader) * param.eval_freq)
    print("evaluate step=====", eval_step)
    optimizer = torch.optim.AdamW(model.parameters(), lr=param.learning_rate)

    train_step = 0
    for epoch in range(start_epoch, param.epoches):
        model.train()
        optimizer.zero_grad()

        train_start = time.time()
        tr_loss = 0.0
        start = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_norm = AverageMeter()

        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            train_step += 1
            data_time.update(time.time() - start)
            for name in batch:
                batch[name] = batch[name].to(device)

            pred_ctr = model(batch)
            # print("pred", pred_ctr.shape, pred_ctr)
            # print("lael", batch['label'].shape, batch['label'])
            loss = criterion(pred_ctr, batch['label'].float())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tr_loss += loss.item()
            losses.update(loss.item())
            try:
                losses_norm.update(norm_loss.item())
            except:
                losses_norm.update(0.0)
            batch_time.update(time.time() - start)
            start = time.time()
            # print("model bert", model.bert.encoder.layer[0].output.dense.weight.requires_grad,
            #       model.bert.encoder.layer[0].output.dense.weight)

            if (step + 1) % param.print_freq == 0:
                print("Epoch:{}-{}/{}, loss: [{}], loss-norm: [{}], [{}], [{}] ".\
                      format(epoch, step, len(train_dataloader), meter_to_str("Loss", losses, RBIT),
                             meter_to_str("Loss norm", losses_norm, RBIT),
                             meter_to_str("Batch_Time", batch_time, RBIT),
                             meter_to_str("Data_Load_Time", data_time, RBIT)))
            # 根据当前训练的轮次，更新优化器：对于需要更新Swin和Bert参数时
            # 可以设置前多少步（train_step）更新预训练好的"大模型"参数，而后进行"冻结"
            if (step + 1) % eval_step == 0 and test_dataloader:
                if isinstance(model, torch.nn.DataParallel):
                    torch.save({'epoch': epoch, 'model_state': model.module.state_dict(), 'loss': tr_loss},
                               os.path.join(param.model_folder, f'checkpoint_{epoch}_{(step + 1)}.pt'))
                else:
                    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'loss': tr_loss},
                           os.path.join(param.model_folder, f'checkpoint_{epoch}_{(step+1)}.pt'))
                evaluate(model, test_dataset, test_dataloader, device, epoch, param, eval_step=(step+1))

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}minute".format(epoch, tr_loss,
                                                                                 (time.time() - train_start) / 60.0))
        if isinstance(model, torch.nn.DataParallel):
            torch.save({'epoch': epoch, 'model_state': model.module.state_dict(), 'loss': tr_loss},
                   os.path.join(param.model_folder, f'checkpoint_{epoch}.pt'))
        else:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'loss': tr_loss},
                   os.path.join(param.model_folder, f'checkpoint_{epoch}.pt'))

        if test_dataloader:
            evaluate(model, test_dataset, test_dataloader, device, epoch, param, eval_step="last")


def evaluate(model, test_dataset, predict_dataloader, device, epoch_th, param, eval_step="last"):
    # print("***** Running prediction *****")
    fw = open(param.outpath + f'-{epoch_th}-{eval_step}', 'w', newline='')
    writer = csv.DictWriter(fw, delimiter=',', fieldnames=test_dataset.need_keys+['cal_score', 'pctr_int', 'pctr', 'label'])
    writer.writeheader()
    model.eval()
    start = time.time()
    criterion = F.binary_cross_entropy  # binary_cross_entropy  # Cross loss

    losses = AverageMeter()
    pred_idx = 0

    with torch.no_grad():
        for step, batch in enumerate(predict_dataloader):
            for name in batch:
                batch[name] = batch[name].to(device)
            pred_ctr = model(batch)

            for one_pred in pred_ctr.tolist():
                one_sample = test_dataset.samples[pred_idx]
                one_sample['cal_score'] = float(one_pred)
                writer.writerow(one_sample)
                pred_idx += 1

            loss = criterion(pred_ctr, batch['label'].float())

            losses.update(loss.item())
            if (step + 1) % param.print_freq == 0:
                print(f"evaluate on epoch={epoch_th}, inter-step={step}/{len(predict_dataloader)}, "
                      f"loss={meter_to_str('Loss', losses, RBIT)}")
    model.train()
    fw.close()


if __name__ == '__main__':
    param = argparse.ArgumentParser(description='Train NLP for ctr Model')
    param.add_argument("--sample-path", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--test-sample-path", type=str,
                       default=r"", help="Data path for testing")
    param.add_argument("--split-path", type=str,
                       default=r"", help="Data path for split pctr json")
    param.add_argument("--cross-path", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--need-keys", type=str,
                       default="", help="Need field names")
    param.add_argument("--label-name", type=str,
                       default="", help="Field name of label")
    param.add_argument("--pctr-label-name", type=str,
                       default="", help="Field name of pctr")
    param.add_argument("--outpath", type=str,
                       default=r"./res.csv", help="Outpath for testing samples")
    param.add_argument("--ts-weight-folder", type=str,
                       default=r"", help="")
    param.add_argument("--checkpoint-path", type=str,
                       default="", help="Pre-trained model path")
    param.add_argument("--batch-size", type=int,
                       default=4, help="Batch size of samples")
    param.add_argument("--workers", type=int,
                       default=0, help="Workers of dataLoader")
    param.add_argument("--epoches", type=int,
                       default=1, help="Epoches")
    param.add_argument("--learning-rate", type=float,
                       default=5e-5, help="Learning rate for model when training")  #TODO
    param.add_argument("--model-folder", type=str,
                       default="./models", help="Folder for saved models")
    param.add_argument("--print-freq", type=int,
                       default=10, help="Frequency for printing training progress")
    param.add_argument("--eval-freq", type=float, default=1.1, help="Percentage of evaluation")

    # model parameters:
    param.add_argument("--emb-size", type=int, default=32, help="embedding size for each field")
    param.add_argument("--dropout", type=float, default=0.2, help="dropout prob for fc-linear")
    param.add_argument("--fc-hidden-size-str", type=str, default='128,64,32,1', help="")
    param.add_argument("--lambda-v", type=float, default=1.0, help="lambda for norm loss")

    param.add_argument("--debug", action='store_true')
    param.add_argument("--cpu", action='store_true')
    param.add_argument("--stop-gradient", action='store_true')
    param.add_argument("--seed", type=int, default=44, help="random seed")

    param = param.parse_args()
    print("Param", param)
    main(param)
