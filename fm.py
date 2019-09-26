import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.sparse import csr
from tqdm import tqdm_notebook as tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

TRAIN_RATIO = 0.9
LATENT_DIM = 10
LAMBDA = 1
LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 1000

def vectorize_dict(dic, dim=None):
    feature_num = len(list(dic.keys()))
    record_num = len(list(dic.items())[0][1])
    col_ix = np.zeros([feature_num*record_num])
    
    ix = {}
    i = 0
    for k in dic.keys():
        lis = dic[k]
        for t in range(len(lis)):
            ix[str(k) + str(lis[t])] = ix.get(str(k) + str(lis[t]), 0) + 1
            col_ix[t*feature_num + i] = ix[str(k) + str(lis[t])]
        i += 1

    # ix = {}
    # i = 0
    # count = 0
    # for k in dic.keys():
    #     lis = dic[k]
    #     for t in range(len(lis)):
    #         flag = str(k) + str(lis[t])
    #         if flag not in ix.keys():
    #             ix[flag] = count
    #             count += 1
    #         col_ix[t*feature_num + i] = ix[flag]
    
    if dim == None: dim = len(ix)
    row_ix = np.repeat(np.arange(0, record_num), feature_num)
    ixx = np.where(col_ix < dim)
    data = np.ones([feature_num*record_num])
    
    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=[record_num, dim]), ix

def batcher(X_, y_, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_X = X_[i:upper_bound]
        ret_y = None
        ret_y = y_[i:upper_bound]
        yield (ret_X, ret_y)

df = pd.read_csv('./data/ml-latest-small/ratings.csv', delimiter=',')
train_num = int(df.shape[0]*TRAIN_RATIO)
x_train, ix = vectorize_dict(
    {'users': df['userId'].values[:train_num], 'items': df['movieId'].values[:train_num]})
x_test, ix = vectorize_dict(
    {'users': df['userId'].values[train_num:], 'items': df['movieId'].values[train_num:]}, dim=len(ix))
y_train = df['rating'].values[:train_num]
y_test = df['rating'].values[train_num:]

x_train = x_train.todense()
x_test = x_test.todense()

n, dim = x_train.shape
k = LATENT_DIM

w0 = tf.Variable(tf.truncated_normal([]))
w = tf.Variable(tf.truncated_normal([1, dim]))
V = tf.Variable(tf.truncated_normal([k, dim]))

x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None])
linear_part = w0 + tf.reduce_sum(x*w, -1)
interactive_part = 0.5*tf.reduce_sum(tf.matmul(x, tf.transpose(V))**2 - tf.matmul(x**2, tf.transpose(V)**2), 1)
y_ = linear_part + interactive_part

target_loss = tf.reduce_mean((y - y_)**2)
l2_norm = tf.nn.l2_loss(w) + tf.nn.l2_loss(V)
loss = target_loss + LAMBDA*l2_norm
error = tf.reduce_mean((y - y_)**2)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in tqdm(range(EPOCHS), unit='epoch'):
    permu = np.random.permutation(x_train.shape[0])
    for b_x, b_y in batcher(x_train[permu], y_train[permu], BATCH_SIZE):
        _, l = sess.run([train_op, loss], {x: b_x, y: b_y})
        print("loss: %.3f"%(l))

errors = []
for b_x, b_y in batcher(x_test, y_test):
    errors.append(sess.run(error, {x: b_x, y: b_y}))
print("rmse %.3f"%((np.mean(errors))**0.5))


