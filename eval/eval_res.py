import numpy as np
import json
import pandas as pd
import math
np.random.seed(42)

pd.options.mode.chained_assignment = None
import csv, sys
from sklearn.metrics import log_loss, roc_auc_score

if len(sys.argv) > 1: inpath = sys.argv[1]
else:
    print("please input the 'res.csv'") 
    sys.exit(-1)
#print("inpath", inpath)
field_value = '121'
if len(sys.argv) > 2: field_value = sys.argv[2]
#print("field_value", field_value)
    
num_shuffles = 10
random_state = [np.random.randint(10000) for _ in range(num_shuffles)]

fid, cid, iid, raw_score, cal_score, labels = [], [], [], [], [], []


with open(inpath, 'r', encoding='utf-8-sig') as f:
    lines = csv.DictReader(f)
    for line in list(lines):
        y = 1 if int(line['label'])>0 else 0
        
        fid.append(float(line[field_value]))
        cid.append(float(line[field_value]))
        iid.append(float(line[field_value]))
        raw_score.append(float(line['pctr']))
        cal_score.append(float(line['cal_score']))
        labels.append(y)

eval_data = {
    'field': np.array(fid),
    'cid': np.array(cid),
    'iid': np.array(iid),
    'raw_score': np.array(raw_score),
    'cal_score': np.array(cal_score),
    'labels': np.array(labels),
}        

#print("#samples", len(labels))


df = pd.DataFrame(eval_data)

logloss = log_loss(np.array(labels), np.array(raw_score)), log_loss(np.array(labels), np.array(cal_score))
auc_value = roc_auc_score(np.array(labels), np.array(raw_score)), roc_auc_score(np.array(labels), np.array(cal_score))
pcoc_value = np.mean(np.array(raw_score)) / np.mean(np.array(labels)), np.mean(np.array(cal_score)) / np.mean(np.array(labels))

def getECE(data, M=10):
    # reference: <<On Calibration of Modern Neural Networks>>
    # https://arxiv.org/pdf/1706.04599.pdf
    data['bucket'] = pd.qcut(data['raw_score'].rank(method='first'), M, labels=False, duplicates="drop")
    q_curve = data.groupby(by='bucket').agg({
        'raw_score': 'mean',
        'cal_score': 'mean',
        'labels': ['mean', 'count'],
        'bucket': 'max'
    })

    if(q_curve.empty):
        return 0, 0

    n = 0
    raw_ece = 0.0
    cal_ece = 0.0


    for row in q_curve.index:
        bm = q_curve.loc[row]['labels']['count']
        raw_ece = raw_ece + abs(q_curve.loc[row]['raw_score']['mean'] - q_curve.loc[row]['labels']['mean']) * bm
        cal_ece = cal_ece + abs(q_curve.loc[row]['cal_score']['mean'] - q_curve.loc[row]['labels']['mean']) * bm
        n = n + bm

    return raw_ece / n, cal_ece / n

def getFieldECE_ori(dim, data, M = 3):
    # uniq feature value set
    dims = data[dim].unique()

    field_raw_ece = 0.0
    field_cal_ece = 0.0

    n = 0
    for d in dims:
        dim_df = data[data[dim] == d]
        cnt = dim_df.shape[0]

        raw_ece, cal_ece = getECE(dim_df, M)
        field_raw_ece = field_raw_ece + raw_ece * cnt 
        field_cal_ece = field_cal_ece + cal_ece * cnt
        n = n + cnt

    return field_raw_ece / n, field_cal_ece / n

def getFieldECE(dim, df, M = 3):
    df = df.sort_values(by=[dim,'raw_score'])
    #print("sorted")
    df['bucket'] = df.groupby(dim)['raw_score'].apply(
        lambda x: pd.qcut(x.rank(method='first'), M, labels=False, duplicates='drop')
    ).values
    #print("bucket")
    q_curve = df.groupby([dim, 'bucket'], as_index=True).agg(
        raw_score_mean = ('raw_score', 'mean'),
        cal_score_mean = ('cal_score', 'mean'),
        labels_mean = ('labels', 'mean'), 
        labels_count = ('labels','count')
    )
    #print("calculated")

    # Rename multi-level columns
#     q_curve.columns = [f"{col[0]}_{col[1]}" if len(col) == 2 else col for col in q_curve.columns]
    
    # Calculate ECE values within the 'q_curve' DataFrame
    q_curve['raw_ece'] = abs(q_curve['raw_score_mean'] - q_curve['labels_mean']) * q_curve['labels_count']
    q_curve['cal_ece'] = abs(q_curve['cal_score_mean'] - q_curve['labels_mean']) * q_curve['labels_count']


    raw, cal = q_curve[['raw_ece', 'cal_ece']].sum() /  q_curve['labels_count'].sum()
    return raw, cal


def getFieldRCE(dim, data):
    filedPd = data.groupby(dim).agg({
        'raw_score': 'sum',
        'cal_score': 'sum',
        'labels': ['sum', 'count'],
    })
    filedPd = filedPd[filedPd['labels']['sum'] > 0]

    n = 0
    raw_field_rce = 0.0
    cal_field_rce = 0.0

    for row in filedPd.index:
        bm = filedPd.loc[row]['labels']['count']
        raw_field_rce = raw_field_rce + \
                        bm * abs(filedPd.loc[row]['raw_score']['sum'] - filedPd.loc[row]['labels']['sum']) / filedPd.loc[row]['labels']['sum']
        cal_field_rce = cal_field_rce + \
                        bm * abs(filedPd.loc[row]['cal_score']['sum'] - filedPd.loc[row]['labels']['sum']) / filedPd.loc[row]['labels']['sum']
        n = n + bm

    return raw_field_rce / n, cal_field_rce / n

def getSampleRCE(data):
    #print("SampleRCE")
    n = len(data)
    raw_error = abs(data['raw_score'] - data['labels']).sum()
    cal_error = abs(data['cal_score'] - data['labels']).sum()

    return raw_error / n, cal_error / n

def getFieldECE_RAW(dim, data, M = 3):
    # uniq feature value set
    dims = data[dim].unique()
    field_raw_ece = 0.0
    field_cal_ece = 0.0

    n = 0
    for d in dims:
        dim_df = data[data[dim] == d]
        raw_ece, cal_ece = getECE(dim_df, M)
        field_raw_ece = field_raw_ece + raw_ece
        field_cal_ece = field_cal_ece + cal_ece
        n = n + 1

    return field_raw_ece / n, field_cal_ece / n

def getMVCE(df, col, label_col, num_partitions = 2000):
    # return abs(df['label'] - df[col]).mean()
    # print()
    mvce = []
    for r in random_state:
        pce = []

        # Shuffle the DataFrame randomly
        df_shuffled = df.sample(frac=1, random_state=r)

        # Split the shuffled DataFrame into equal-sized partitions
        partitions = np.array_split(df_shuffled, num_partitions)

        for partition in partitions:
            label_avg = partition[label_col].mean()
            pctr_avg = partition[col].mean()

            # Calculate and store the result for each partition
            pce.append(abs(label_avg - pctr_avg))
        mvce.append(np.mean(pce) * np.mean(pce))
        # print(np.mean(pce) * np.mean(pce))
    return (np.sqrt(np.mean(mvce)))

result = {
    'logloss': logloss,
    'AUC': auc_value,
    'PCOC': pcoc_value,
    'F-RCE'  : getFieldRCE("cid", df),
    'S-Error': getSampleRCE(df),
    # 'ECE'    : getECE(df),
    'F-ECE@3': getFieldECE_RAW('cid', df),
    'F-W-ECE@3'  : getFieldECE('cid', df),
    'F-ECE@5': getFieldECE_RAW('cid', df, 5),
    'F-W-ECE@5': getFieldECE('cid', df, 5),
    'F-ECE@10': getFieldECE_RAW('cid', df, 10),
    'F-W-ECE@10': getFieldECE('cid', df, 10),
    'MVCE': [getMVCE(df, "raw_score", "labels"), getMVCE(df, "cal_score", "labels")],
}
print("result", json.dumps(result, indent=2))
