#!/usr/bin/env python
# coding=utf8


import os
import time
import torch
import math
import numpy as np

def compute_norm_loss(b_param):
    norm_loss = 0.0
    for x in range(len(b_param) - 1):
        delta_loss = max(0.0, b_param[x] - b_param[x + 1])
        norm_loss += delta_loss
    return norm_loss

def get_opt(model, param):
    if param.fix_swin:
        for m in model.swin_model.parameters():
            m.requires_grad = False
    if param.fix_bert:
        for m in model.bert_model.parameters():
            m.requires_grad = False

    param_optimizer = list(model.named_parameters())

    swin_param = ['swin_model']
    bert_param = ['bert_model']
    swin_param_opt = [p for n, p in param_optimizer if any(nd in n for nd in swin_param)]
    bert_param_opt = [p for n, p in param_optimizer if any(nd in n for nd in bert_param)]
    base_param_opt = [p for n, p in param_optimizer if not any(nd in n for nd in swin_param) and \
                      not any(nd in n for nd in bert_param)]
    print("opt parameter size", len(swin_param_opt), len(bert_param_opt), len(base_param_opt))

    # optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate)  # , warmup=param.warmup_proportion, t_total=total_train_steps
    if param.fix_swin and param.fix_bert:
        optimizer_grouped_parameters = [{'params': base_param_opt}]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=param.learning_rate)  # , warmup=param.warmup_proportion, t_total=total_train_steps
        optimizer.param_groups[0]['lr'] = param.learning_rate  # base lr
        print("only optimize dense base parameters")
    elif not param.fix_swin and param.fix_bert:
        optimizer_grouped_parameters = [{'params': swin_param_opt}, {'params': base_param_opt}]
        # total_train_steps = int(train_dataset.__len__() / param.batch_size / param.gradient_accumulation_steps * param.epoches)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=param.learning_rate)  # , warmup=param.warmup_proportion, t_total=total_train_steps
        optimizer.param_groups[0]['lr'] = param.learning_rate_swin  # swin lr
        optimizer.param_groups[1]['lr'] = param.learning_rate  # dense base lr
        print("optimize swin+base")
    elif param.fix_swin and not param.fix_bert:
        optimizer_grouped_parameters = [{'params': bert_param_opt}, {'params': base_param_opt}]
        # total_train_steps = int(train_dataset.__len__() / param.batch_size / param.gradient_accumulation_steps * param.epoches)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=param.learning_rate)  # , warmup=param.warmup_proportion, t_total=total_train_steps
        optimizer.param_groups[0]['lr'] = param.learning_rate_bert  # swin lr
        optimizer.param_groups[1]['lr'] = param.learning_rate  # dense base lr
        print("optimize bert+base")
    else:
        optimizer_grouped_parameters = [{'params': swin_param_opt}, {'params': bert_param_opt},
                                        {'params': base_param_opt}]
        # total_train_steps = int(train_dataset.__len__() / param.batch_size / param.gradient_accumulation_steps * param.epoches)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=param.learning_rate)  # , warmup=param.warmup_proportion, t_total=total_train_steps
        optimizer.param_groups[0]['lr'] = param.learning_rate_swin  # swin lr
        optimizer.param_groups[1]['lr'] = param.learning_rate_bert  # swin lr
        optimizer.param_groups[2]['lr'] = param.learning_rate  # dense base lr
        print("optimize swin+bert+base")
    return optimizer


def update_opt(model, optimizer, param, train_step):
    if not param.fix_swin and train_step == param.num_step_swin:
        print(f"---swin_model fixed on {train_step}")
        if isinstance(model, torch.nn.DataParallel):
            for m in model.module.swin_model.parameters():
                m.requires_grad = False
        else:
            for m in model.swin_model.parameters():
                m.requires_grad = False
        optimizer.param_groups[0]['lr'] = 0.0
    if not param.fix_bert and train_step == param.num_step_bert:
        print(f"---bert_model fixed on {train_step}")
        if isinstance(model, torch.nn.DataParallel):
            for m in model.module.bert_model.parameters():
                m.requires_grad = False
        else:
            for m in model.bert_model.parameters():
                m.requires_grad = False
        if param.fix_swin:
            optimizer.param_groups[0]['lr'] = 0.0
        else:
            optimizer.param_groups[1]['lr'] = 0.0


def compute_rig(y, pred_prob, mean_p):
    N = len(y)
    fz = 0
    for i in range(len(y)):
        fz += ((1 + y[i]) * math.log10(pred_prob[i]) + (1 - y[i]) * math.log10(1 - pred_prob[i])) / 2
    fz = -(fz / N)
    fm = -(mean_p * math.log10(mean_p) + (1 - mean_p) * math.log10(1 - mean_p))
    return 1 - fz / fm


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, val_n=1):
        self.val = val
        self.sum += val * val_n
        self.count += val_n
        self.avg = self.sum / self.count


def meter_to_str(info_title, avg_meter, round_bit):
    return "{} {}({})\t".format(info_title,
                                round(avg_meter.val, round_bit), round(avg_meter.avg, round_bit))


def save_checkpoint(epoch, file_index, model, model_optimizer, model_folder):
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    filename = 'checkpoint-{}-{}.pth.tar'.format(epoch, file_index)
    opt_filename = 'opt-{}-{}.pth.tar'.format(epoch, file_index)
    for _ in range(5):
        try:
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(model_folder, filename))
            else:
                torch.save(model.state_dict(), os.path.join(model_folder, filename))
            torch.save(model_optimizer.state_dict(), os.path.join(model_folder, opt_filename))
            break
        except:
            time.sleep(5)


def save_checkpoint_ori(epoch, file_index, model, model_optimizer, model_folder):
    state = {'epoch': epoch,
             'model': model,
             'model_optimizer': model_optimizer}
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    filename = 'checkpoint-{}-{}.pth.tar'.format(epoch, file_index)
    for _ in range(5):
        try:
            torch.save(state, os.path.join(model_folder, filename))
            break
        except:
            time.sleep(5)


def load_model(checkpoint_path, model):
    if not os.path.exists(checkpoint_path):
        print("No input model {}".format(checkpoint_path))
        return model
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), True)
    except Exception as err:
        print("Error in load model", err)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), True)
    if hasattr(model, "module"):
        model = model.module  # 如果是多GPU的DataParallel格式则使用model.module，否则（单gpu）则去掉该行
    return model


def load_optimizer(checkpoint_path, optimizer):
    if not os.path.exists(checkpoint_path):
        print("No input optimizer {}".format(checkpoint_path))
        return optimizer
    optim_recover = torch.load(checkpoint_path, map_location='cpu')
    if hasattr(optim_recover, 'state_dict'):
        optim_recover = optim_recover.state_dict()
    optimizer.load_state_dict(optim_recover)
    return optimizer


def load_model_ori(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = checkpoint['model']
    if hasattr(model, "module"):
        model = model.module  # 如果是多GPU的DataParallel格式则使用model.module，否则（单gpu）则去掉该行
    return model


def load_model_and_transform(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = checkpoint['model']
    if hasattr(model, "module"):
        model = model.module  # 如果是多GPU的DataParallel格式则使用model.module，否则（单gpu）则去掉该行
    transform = checkpoint["transform"]
    return model, transform


def compute_cos_sim(vec1, vec2):
    return float(np.sum(vec1 * vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def compute_pairwise_loss(pred_tensor, target_tensor, weight_tensor, loss_func, topk=2, topk_weight=(0.8, 0.2)):
    """
    :param pred_tensor: bs, #creatives
    :param target_tensor: bs, #creatives
    :param weight_tensor: bs, #creatives
    :param topk: int
    :return:
    """

    def index_select(tensor, dim, index):
        return tensor.gather(dim, index.unsqueeze(dim)).squeeze(dim)

    target_vs, indices = target_tensor.topk(topk, dim=-1)  # max, max-1, ...
    _, indices_neg = (-target_tensor).topk(target_tensor.shape[1] - topk, dim=-1)  # min, min+1, ...
    # print("indices", indices)
    # print("indices_neg", indices_neg)

    pred_tensor_exp = pred_tensor.unsqueeze(1).repeat(1, topk, 1)
    weight_tensor_exp = weight_tensor.unsqueeze(1).repeat(1, topk, 1)
    pred_vs = index_select(pred_tensor_exp, 2, indices)
    weight_vs = index_select(weight_tensor_exp, 2, indices)
    pred_tensor_exp_neg = pred_tensor.unsqueeze(1).repeat(1, target_tensor.shape[1] - topk, 1)
    weight_tensor_exp_neg = weight_tensor.unsqueeze(1).repeat(1, target_tensor.shape[1] - topk, 1)
    pred_vs_neg = index_select(pred_tensor_exp_neg, 2, indices_neg)
    weight_vs_neg = index_select(weight_tensor_exp_neg, 2, indices_neg)
    # print("pred_vs", pred_vs)
    # print("pred_vs_neg", pred_vs_neg)
    #
    # print("weight_vs", weight_vs)
    # print("weight_vs_neg", weight_vs_neg)

    all_x1 = []
    all_x2 = []
    all_y = []
    all_w = []
    for idx in range(pred_vs.shape[0]):
        one_pos = pred_vs[idx, :]  # topk pos samples
        one_neg = pred_vs_neg[idx, :]  # neg samples
        one_w_pos = weight_vs[idx, :]
        one_w_neg = weight_vs_neg[idx, :]

        # print("---", one_pos.shape, one_neg.shape, one_w_pos.shape, one_w_neg.shape)
        for jdx in range(one_pos.shape[0]):
            for kdx in range(one_neg.shape[0]):
                all_x1.append(float(one_pos[jdx]))
                all_x2.append(float(one_neg[kdx]))
                wei1 = one_w_pos[jdx]
                wei2 = one_w_neg[kdx]
                if wei1 != 0 and wei2 != 0:
                    all_y.append(1)
                else:
                    all_y.append(0)
                all_w.append((wei1 + wei2) / 2)

    all_x1 = torch.FloatTensor(all_x1).to(pred_tensor.device)
    all_x2 = torch.FloatTensor(all_x2).to(pred_tensor.device)
    all_w = torch.FloatTensor(all_w).to(pred_tensor.device)
    all_y = torch.LongTensor(all_y).to(pred_tensor.device)
    loss = torch.mean(loss_func(all_x1, all_x2, all_y) * all_w)
    return loss

