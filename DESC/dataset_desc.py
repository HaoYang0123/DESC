import csv, os
import bisect
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
try:
    from transformers import AutoImageProcessor, SwinForMaskedImageModeling
except: pass

class CalXXXDataset(Dataset):
    def __init__(self, sample_path, split_path, cross_path, cross_value={}, need_keys=[], label_name="", pctr_label_name="", debug=False, train_flag=True):
        self.need_keys = need_keys #['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207',
                          #'210', '216', '508', '509', '702', '853', '301', '109_14', '110_14', '127_14', '150_14']
        self.label = label_name  #"click"/"label"
        self.pctr_label = pctr_label_name  #"pctr"/"raw_score"
        self.max_value = {key: 0 for key in self.need_keys}
        self.cross_feats = self._load_cross(cross_path)
        for (feat1, feat2) in self.cross_feats:
            self.max_value[str(feat1)+"@"+str(feat2)] = 0
            self.need_keys.append(str(feat1)+"@"+str(feat2))
        self.pctr_list = self._load_pctr_json(split_path)
        print("--->>> cross_value", len(cross_value))
        self.samples, self.crossfield2value = self._load_sample(sample_path, debug, cross_value, train_flag)
        if debug:
            self.samples = self.samples[:100]

    def __len__(self):
        return len(self.samples)

    def _load_pctr_json(self, split_path):
        with open(split_path) as f:
            d = json.load(f)
        max_v = float(d[-1]) + 1e-3
        return [float(v) for v in d] + [max_v]
    
    def _load_cross(self, cross_path):
        if not os.path.exists(cross_path):
            return []
        cross_feats = []
        with open(cross_path) as f:
            for line in f:
                tmp = line.strip('\n').split(",")
                if len(tmp) != 2: 
                    print("[Warning] cross features size != 2", tmp)
                    continue
                cross_feats.append([tmp[0], tmp[1]])
        return cross_feats

    def __getitem__(self, item_idx):
        return self.samples[item_idx]

    def _load_sample(self, sample_path, debug, crossfield2value, train_flag):
        samples = []

        with open(sample_path, 'r', encoding='utf-8-sig') as f:
            lines = csv.DictReader(f)
            for lidx, line in enumerate(lines):
                if lidx % 100000 == 0:
                    print(lidx)
                one_sample = {}
                for key in self.need_keys:
                    if key.find("@") >= 0: continue
                    one_sample[key] = int(line[key])
                    self.max_value[key] = max(self.max_value[key], int(line[key]))
                for (feat1, feat2) in self.cross_feats:
                    cfield = str(feat1) + "@" + str(feat2)
                    cvalue = str(line[feat1]) + "@" + str(line[feat2])
                    if cfield not in crossfield2value:
                        if not train_flag:
                            print("[Warning] not has this cross feature in testing data", cfield)
                        crossfield2value[cfield] = {}
                    if train_flag:
                        if cvalue not in crossfield2value[cfield]: 
                            crossfield2value[cfield][cvalue] = len(crossfield2value[cfield]) + 1
                    else:
                        if cvalue not in crossfield2value[cfield]: 
                            crossfield2value[cfield][cvalue] = 0
                            print("[Warning] not has this cross feature-value in testing data", cfield, cvalue)
                    one_sample[cfield] = crossfield2value[cfield][cvalue]
                    self.max_value[cfield] = max(self.max_value[cfield], one_sample[cfield])
                one_sample['label'] = 1 if int(line[self.label]) > 0 else 0
                one_sample['pctr'] = float(line[self.pctr_label])
                # print("--->>", self.pctr_list, one_sample['pctr'])
                pctr_int = bisect.bisect_right(self.pctr_list, one_sample['pctr'])-1
                one_sample['pctr_int'] = pctr_int
                samples.append(one_sample)
                if debug and len(samples) >= 100:
                    break
        return samples, crossfield2value

if __name__ == '__main__':
    sample_path = "/Users/apple.yang/Documents/Data/Calibration/debug.csv"
    split_path = "/Users/apple.yang/Documents/Data/Calibration/pctr_split.json"

    dataset = CalDataset(sample_path=sample_path, split_path=split_path)
    # dataload = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    # for batch in dataload:
    #     print(batch)