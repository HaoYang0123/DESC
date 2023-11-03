import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertConfig, BertModel
import math

from attentions import MultiHeadAttention


class CalXXXModel(nn.Module):
    def __init__(self, emb_size=32, bin_name2bin_num={}, field_name_list=[],
                 fc_hidden_size=[128, 64, 32, 1], globle_hidden_size=64, debias_hidden_size=8, drop_prob=0.2, 
                 field_atten_dim=[200,200], ts_list_b=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], ts_list=[1.0, 1.2, 0.8, 0.96, 1.4, 1.68, 1.12, 1.3439999999999999, 0.6, 0.72, 0.48, 0.576, 0.8399999999999999, 1.008, 0.672, 0.8063999999999999], stop_gradient=False, ts_weight_folder=''):
        super(CalXXXModel, self).__init__()
        self.stop_gradient = stop_gradient
        self.emb_size = emb_size
        self.bin_name2bin_num = bin_name2bin_num
        self.field_name_list = field_name_list
        
#         if not os.path.exists(ts_weight_folder):
#             print("not input ts_weight_folder")
#         self.ts_weight = self._load_ts_weight_info(ts_weight_folder)  # #curves, 48+1, first 48 are weights, last 1 is bias
#         print("----ts_weight----------", self.ts_weight.shape)
        
        #self.field_name_list = ["124", "128", "301", "126", "127", "129", "125", "122", "121", "109_14", "508", "206"]  # , "150_14", "853"   # very bad

        self.field_embeddings = nn.ModuleList([
            nn.Embedding(self.bin_name2bin_num[field_name]+1, self.emb_size) for field_name in self.field_name_list
        ])
        self.pctr_embedding = nn.Embedding(102, self.emb_size)
        for one_emb in self.field_embeddings:
            nn.init.xavier_uniform_(one_emb.weight)
        nn.init.xavier_uniform_(self.pctr_embedding.weight)

        self.final_dim_size = len(self.field_name_list) * emb_size + emb_size

#         self.ctr_fc = nn.Sequential(
#             nn.Linear(self.final_dim_size, fc_hidden_size[0]),
#             nn.BatchNorm1d(fc_hidden_size[0]),
#             nn.ReLU(),
#             nn.Dropout(p=drop_prob),
#             nn.Linear(fc_hidden_size[0], fc_hidden_size[1]),
#             nn.BatchNorm1d(fc_hidden_size[1]),
#             nn.ReLU(),
#             nn.Dropout(p=drop_prob),
#             nn.Linear(fc_hidden_size[1], fc_hidden_size[2]),
#             nn.BatchNorm1d(fc_hidden_size[2]),
#             nn.ReLU(),
#             nn.Dropout(p=drop_prob),
#             nn.Linear(fc_hidden_size[2], fc_hidden_size[3]),
#         )
        
        self.field_atten_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size * 2, field_atten_dim[0]),
                # nn.BatchNorm1d(field_atten_dim[0]),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(field_atten_dim[0], field_atten_dim[1]),
                # nn.BatchNorm1d(field_atten_dim[1]),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(field_atten_dim[1], len(ts_list)*3),  # len(ts_list)*3
                nn.Softmax(),
            )
            for _ in self.field_name_list
            ] + [nn.Sequential(
                nn.Linear(len(self.field_name_list) * emb_size + emb_size, field_atten_dim[0]),
                # nn.BatchNorm1d(field_atten_dim[0]),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(field_atten_dim[0], field_atten_dim[1]),
                # nn.BatchNorm1d(field_atten_dim[1]),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(field_atten_dim[1], len(ts_list)*3),  # len(ts_list)*3
                nn.Softmax(),
        )])
        self.field_up_fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(ts_list)*3, len(ts_list)*3//2),  # len(ts_list)*3
                # nn.BatchNorm1d(10),
                nn.Sigmoid(),
                # nn.ReLU(),
                nn.Linear(len(ts_list)*3//2, 1),
            )
            for _ in range(len(self.field_name_list)+1)
        ])
        
        self.ts_list_b = ts_list_b
        self.ts_list = ts_list
        
        # TODO, fc_hidden_size not used
        
        self.global_fc0 = nn.Sequential(
            nn.Linear(self.final_dim_size, globle_hidden_size),
            nn.ReLU(),
        )
        self.global_fc1 = nn.Sequential(
            nn.Linear(globle_hidden_size, len(self.field_name_list) + 1),
            nn.Softmax(),
        )
        self.debias = nn.Sequential(
            nn.Linear(globle_hidden_size, debias_hidden_size),
            nn.ReLU(),
            nn.Linear(debias_hidden_size, 1),
            nn.Sigmoid(),
        )
        
#         multi_bert_config = BertConfig()
#         multi_bert_config.intermediate_size = emb_size * 4
#         multi_bert_config.num_hidden_layers = 2
#         multi_bert_config.num_attention_heads = 2
#         multi_bert_config.hidden_size = emb_size
#         #减小显存量
#         multi_bert_config.max_position_embeddings = 1
#         multi_bert_config.vocab_size = 1
#         print("multi_bert config", multi_bert_config)
#         self.multi_transformer = BertModel(multi_bert_config)

        self.multi_transformer = MultiHeadAttention(d_model=emb_size, num_heads=8)  #4
    
    def _load_ts_weight_info(self, ts_weight_folder):
        ts_weight_list = []
        for path in os.listdir(ts_weight_folder):
            if not path.endswith('.json'): continue
            with open(os.path.join(ts_weight_folder, path)) as f:
                tmp = json.load(f)
            ts_weight_list.append(tmp)
        return torch.FloatTensor(ts_weight_list)
    
    def _forward_one_field(self, pctr, emb, idx):
        attn_score = self.field_atten_fc[idx](emb)  # bs, 2*32 --> bs, (3*16)
        
        logits = -torch.log(1/pctr - 1)
        #input_x_ensemble = torch.stack([logit * i for i in self.ts_list], dim=1).float()  # bs, 16
        #sca_pctr_ori = torch.stack([1/(1+torch.exp(-1*logits*i)) for i in self.ts_list], dim=1).float()  
        
        pow_pctr = torch.stack([torch.pow(pctr, i) for i in self.ts_list_b], dim=1).float()
        log_pctr = torch.stack([torch.log2(pctr * i + 1) for i in self.ts_list_b], dim=1).float()
        sca_pctr = torch.stack([1/(1+torch.exp(-1*logits*i)) for i in self.ts_list_b], dim=1).float()
        
        input_x_ensemble = torch.cat([pow_pctr, log_pctr, sca_pctr], dim=-1)   # bs, (3*16)
        # input_x_ensemble = torch.stack([torch.sum(input_x_ensemble_ts*curve[:-1], dim=-1) + curve[-1] for curve in self.ts_weight], dim=1).float()  # bs, #curves
        
        # print("input_x_ensemble", input_x_ensemble.shape, input_x_ensemble[:3, :])
        
        #input_x_ensemble = sca_pctr_ori   # bs, 16
        
        input_x_ensemble = input_x_ensemble * attn_score  # bs, (3*16)
        score = self.field_up_fc[idx](input_x_ensemble)  # bs, (16*3)--> bs, 10--> sigmoid--> bs, 1
        return torch.clip(score, min=1e-9, max=1-1e-9)

    def forward_field_all(self, all_emb):
        mid_emb = self.global_fc0(all_emb)  # bs, mid_num
        globle_atten = self.global_fc1(mid_emb)  # bs, 16
        global_debias = self.debias(mid_emb)  # bs
        return globle_atten, global_debias.squeeze(-1)
    
    def forward(self, batch):
        """
        batch is dict:{}
        """
        pctr_int = batch['pctr_int']
        pctr = batch['pctr']
        #logit = -torch.log(1/pctr - 1)
        
        single_emb_list = []
        score_list = []
        pctr_int_emb = self.pctr_embedding(pctr_int)
        for idx, field_name in enumerate(self.field_name_list):  # k个
            v = batch[field_name]  # template_id: bs
            field_emb = self.field_embeddings[idx](v)
            single_emb_list.append(field_emb)  # bs, 32
                           
        field_emb = torch.stack(single_emb_list, dim=1)  # bs, k, 32
        
#         multi_mask = torch.ones((field_emb.shape[0], field_emb.shape[1])).long()
#         multi_attention_mask = multi_mask.unsqueeze(1).unsqueeze(2)
#         multi_attention_mask = multi_attention_mask.to(dtype=field_emb.dtype)  # fp16 compatibility
#         multi_attention_mask = (1.0 - multi_attention_mask) * -10000.0
#         multi_attention_mask = multi_attention_mask.to(field_emb.device)
        field_emb_ext, _ = self.multi_transformer(field_emb, field_emb, field_emb, stop_gradient_flag=self.stop_gradient)  # bs, k, 32
        # print("---->>>", field_emb_ext.shape)
        
        single_emb_ext_list = []
        for idx, field_name in enumerate(self.field_name_list):  # k个
            one_field_emb_ext = field_emb_ext[:, idx, :]  # bs, 32
            single_emb_ext_list.append(one_field_emb_ext)
            #single_emb_ext_list.append(field_emb[:, idx, :])
            emb = torch.cat([pctr_int_emb, one_field_emb_ext], dim=-1)  # bs, 2*32
            one_score = self._forward_one_field(pctr, emb, idx)
            # print("idx", idx, one_score.shape)
            score_list.append(one_score)  
        
        single_emb_ext_list.append(pctr_int_emb)
        all_field_emb = torch.cat(single_emb_ext_list, dim=-1)  # bs, (k+1)*32
        one_score = self._forward_one_field(pctr, all_field_emb, len(self.field_name_list))
        score_list.append(one_score)
        
        # print("scores", len(score_list))
        pred_shape_scores = torch.cat(score_list, dim=-1)  # bs, k
        globle_atten, global_debias = self.forward_field_all(all_field_emb)
        
        pred_shape_final = torch.sum(pred_shape_scores * globle_atten, dim=-1)  # bs
        
        pred_shape_final = pred_shape_final * global_debias  # bs
        
        return pred_shape_final