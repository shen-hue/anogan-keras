import numpy as np
import pandas as pd

test = np.load('result_credit_NN/credit_test_qurey.npy')
test_pred = np.load('result_credit_NN/credit_test_pred.npy')
rec_err = np.linalg.norm(test-test_pred, axis=1)
idx = list(rec_err).index(max(rec_err))
df = pd.DataFrame(data=test_pred[idx], index= range(28),columns=['reconstruction_loss'])

def sort_by_absolute(df, index):
    df_abs = df.apply(lambda x: abs(x))
    df_abs = df_abs.sort_values('reconstruction_loss', ascending=False)
    df = df.loc[df_abs.index, :]
    return df

sort_by_absolute(df, idx).T

top_5_features = sort_by_absolute(df, idx).iloc[:5,:]