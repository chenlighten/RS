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

def csv2dic():
    df = pd.read_csv("data/ml-latest-small/ratings.csv")
    train_num = int(TRAIN_RATIO*(df.shape[0]))

    train_dic = {}
    test_dic = {}
    train_dic['users'] = df['userId'].values[:train_num]
    train_dic['items'] = df['movieId'].values[:train_num]
    train_y = df['rating'].values[:train_num]
    test_dic['users'] = df['userId'].values[train_num:]
    test_dic['items'] = df['movieId'].values[train_num:]
    test_y = df['rating'].values[train_num:]
    
    df = pd.read_csv('data/ml-latest-small/movies.csv')
    gens = []

    def add_time_and_gen(d):
        d['time'] = []
        d['genres'] = []
        for movieId in d['items']:
            where = np.where(df['movieId'].values == movieId)
            if where[0].shape[0] != 0:
                i = where[0][0]
                title = df['title'][i]
                if title[-1] == ')' and title[-6] == '(':
                    time = int(title[-5:-1])
                else:
                    time = 0
                gen = df['genres'][i]
            else:
                time = 0
                gen = '(no genres listed)'
            for g in gen.split('|'):
                if g not in gens: gens.append(g)
            d['time'].append(time)
            d['genres'].append(gen)

    add_time_and_gen(train_dic)
    add_time_and_gen(test_dic)
    return (train_dic, train_y, test_dic, test_y, gens)

def vectorize_dict(dic, gens, dim=None):
    feature_num = len(list(dic.keys()))
    record_num = len(list(dic.values())[0])
    gap = feature_num - 1 + len(gens)
    col_ix = []
    row_ix = []
    
    ix = {}
    for k in dic.keys():
        if k == 'genres':
            before_genres = 0
            lis = dic[k]
            for t in range(len(lis)):
                its_gens = lis[t].split('|')
                for gen in its_gens:
                    ix[str(k) + str(gen)] = ix.get(str(k) + str(gen), 0) + 1
                    col_ix.append(ix[str(k) + str(gen)])
                    row_ix.append(t)
        else:
            lis = dic[k]
            for t in range(len(lis)):
                ix[str(k) + str(lis[t])] = ix.get(str(k) + str(lis[t]), 0) + 1
                col_ix.append(ix[str(k) + str(lis[t])])
                row_ix.append(t)

    col_ix = np.array(col_ix)
    row_ix = np.array(row_ix)

    # ix = {}
    # i = 0
    # count = 0
    # for k in dic.keys():
    #     lis = dic[k]
    #     for t in range(len(lis)):
    #         flag = str(k) + str(lis[t])
    #         if flag not in ix.keys():gap*record_num
    #             ix[flag] = count
    #             count += 1
    #         col_ix[t*feature_num + i] = ix[flag]
    
    if dim == None: dim = len(ix)
    ixx = np.where((col_ix < dim) * (col_ix >= 0))
    data = np.ones([len(col_ix)])
    
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

# df = pd.read_csv('./data/ml-latest-small/ratings.csv', delimiter=',')
# train_num = int(df.shape[0]*TRAIN_RATIO)
# x_train, ix = vectorize_dict(
#     {'users': df['userId'][:train_num], 'items': df['movieId'][:train_num]})
# x_test, ix = vectorize_dict(
#     {'users': df['userId'][train_num:], 'items': df['movieId'][train_num:]}, dim=len(ix))
# y_train = df['rating'][:train_num]
# y_test = df['rating'][train_num:]

train_dic, y_train, test_dic, y_test, gens = csv2dic()
x_train, ix = vectorize_dict(train_dic, gens)
x_test, ix = vectorize_dict(test_dic, gens, dim=len(ix))

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


