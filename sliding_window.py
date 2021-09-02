import numpy as np

def unroll(data,seq_len=100,stride=10,labels=[]):
    un_data = []
    un_labels = []
    idx = 0
    while(idx<(len(data)-seq_len)):
        un_data.append(data[idx:idx+seq_len])
        if len(labels):
            un_labels.append(labels[idx:idx+seq_len])
        idx += stride
    if len(labels):
        return np.array(un_data), np.array(un_labels)
    return np.array(un_data)

def assign_ano(anos):
    y = []
    for ano in anos:
        if 1 in ano:
            y.append(1)
        else:
            y.append(0)
    return np.array(y).reshape(-1,1)
