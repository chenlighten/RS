from datareader import *
from dfm import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

fname = 'data/ml-latest-small/ratings.csv'
fd = FeatureDictionary(fname, fname, ignore_cols=['rating', 'timestamp'])
fd.gen_feat_dict()

dp = DataParser(fd)
Xi, Xv, y = dp.parse(fname, has_label=True)

dfm = DFM(fd.feat_dim, fd.field_dim, loss_type='mse', verbose=1)

train_num = int(len(Xi) * 0.9)
Xi_train, Xv_train, y_train = Xi[:train_num], Xv[:train_num], y[:train_num]
Xi_test, Xv_test, y_test = Xi[train_num:], Xv[train_num:], y[train_num:]

dfm.fit(Xi_train, Xv_train, y_train, Xi_test, Xv_test, y_test)