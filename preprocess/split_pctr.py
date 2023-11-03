import csv
import json

path_list=[
'train.csv',
'test.csv',
#'debug.csv',
]
outpath = 'pctr_split.json'
num = 100

ctr_list = []
for path in path_list:
    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = csv.DictReader(f)
        for line in list(lines):
            pctr = float(line['pctr'])
            ctr_list.append(pctr)

ctr_list.sort()
final_ctr = []

for idx in range(num+1):
    fidx = int(len(ctr_list)*(idx/num))
    fidx = min(len(ctr_list)-1, max(0, fidx))
    final_ctr.append(ctr_list[fidx])


with open(outpath, 'w') as fw:
    json.dump(final_ctr, fw, indent=2)

            
