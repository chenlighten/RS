import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

TRAIN_RATIO = 0.9
K = 10
MAX_RECORDS = 100000
THRESHOLD = 3.5
GAMMA = 25
FILE_NAME = './data/ml-latest-small/ratings.csv'
# FILE_NAME = './data/demo_data'
# FILE_NAME = './data/ml_small_processed'

def dis(user_rating, data):
    return np.sum((data - user_rating) ** 2, axis=-1)

def norm(x):
    return (np.sum(x ** 2)) ** 0.5

def cos_value(x, y):
    index = (x != 0) * (y != 0)
    x = x[index]
    y = y[index]
    return np.dot(x, y) / (norm(x) * norm(y))

def sim_cos(user_rating, data):
    return np.array([cos_value(user_rating, data[i]) for i in range(data.shape[0])])

def cov(x, y):
    return np.dot(x - np.mean(x), y - np.mean(y))

def sim_pearson(user_rating, data):
    res = []
    for i in range(data.shape[0]):
        x = user_rating
        y = data[i]
        index = (x != 0) * (y != 0)
        x = x[index]
        y = y[index]
        res.append(cov(x, y) / (np.std(x) * np.std(y)))
    return np.array(res)

def sim_pearson_importance(user_rating, data):
    res = []
    for i in range(data.shape[0]):
        x = user_rating
        y = data[i]
        index = (x != 0) * (y != 0)
        x = x[index]
        y = y[index]
        res.append(cov(x, y) / (np.std(x) * np.std(y)) * min(len(index), GAMMA) / GAMMA)
    return np.array(res)

def predict_rating_user_cf(user_rating, item_id, data):
    data = data[data[:, item_id] != 0]
    if data.shape[0] == 0: return 0
    sim = sim_pearson_importance(user_rating, data)
    indices = np.argsort(sim)[::-1]
    rating_sum = 0.0
    for i in range(min(K, len(indices))):
        rating_sum += data[indices[i]][item_id]
    return rating_sum / min(K, len(indices))

def predict_rating_user_cf_weight(user_rating, item_id, data):
    data = data[data[:, item_id] != 0]
    if data.shape[0] == 0: return 0
    sim = sim_cos(user_rating, data)
    indices = np.argsort(dis)[::-1]
    rating_sum = 0.0
    weight_sum = 0.0
    for i in range(min(K, len(indices))):
        rating_sum += data[indices[i]][item_id] * sim[indices[i]]
        weight_sum += np.abs(sim[indices[i]])
    return rating_sum / weight_sum

def predict_rating_user_cf_normalized(user_rating, item_id, data):
    data = data[data[:, item_id] != 0]
    if data.shape[0] == 0: return 0

    user_mean = np.mean(user_rating)
    user_std = np.std(user_rating)
    user_rating = (user_rating - user_mean) / user_std
    data = (data - np.mean(data, axis=1).reshape([-1, 1])) / np.std(data, axis=1).reshape([-1, 1])

    dis = np.sum((data - user_rating) ** 2, axis=-1)
    indices = np.argsort(dis)
    rating_sum = 0.0
    for i in range(min(K, len(indices))):
        rating_sum += data[indices[i]][item_id]
    rating = rating_sum / min(K, len(indices))
    return rating * user_std + user_mean

def predict_rating_user_cf_normalized_weight(user_rating, item_id, data):
    data = data[data[:, item_id] != 0]
    if data.shape[0] == 0: return 0

    user_mean = np.mean(user_rating)
    user_std = np.std(user_rating)
    user_rating = (user_rating - user_mean) / user_std
    data = (data - np.mean(data, axis=1).reshape([-1, 1])) / np.std(data, axis=1).reshape([-1, 1])

    dis = np.sum((data - user_rating) ** 2, axis=-1)
    indices = np.argsort(dis)
    rating_sum = 0.0
    weight_sum = 0.0
    for i in range(min(K, len(indices))):
        rating_sum += data[indices[i]][item_id] / (dis[indices[i]]) ** 0.5
        weight_sum += 1 / (dis[indices[i]]) ** 0.5
    return rating_sum / weight_sum

def predict_rating_item_cf(item_rating, user_id, data):
    data = data[:, data[user_id, :] != 0]
    if data.shape[1] == 0: return 0
    dis = np.sum(data - np.reshape(item_rating, [-1, 1]), axis=0)
    indices = np.argsort(dis)
    rating_sum = 0.0
    for i in range(min(K, len(indices))):
        rating_sum += data[user_id][indices[i]]
    return rating_sum / min(K, len(indices))

def predict_rating_item_cf_normalized(item_rating, user_id, data):
    data = data[:, data[user_id, :] != 0]
    if data.shape[1] == 0: return 0

    item_mean = np.mean(item_rating)
    item_std = np.std(item_rating)
    item_rating = (item_rating - item_mean) / item_std
    data = (data - np.mean(data, axis=0).reshape([1, -1])) / np.std(data, axis=0).reshape([1, -1])

    dis = np.sum(data - np.reshape(item_rating, [-1, 1]), axis=0)
    indices = np.argsort(dis)
    rating_sum = 0.0
    for i in range(min(K, len(indices))):
        rating_sum += data[user_id][indices[i]]
    rating = rating_sum / min(K, len(indices))
    return rating * item_std + item_mean

def test_user_cf():
    if FILE_NAME.split('.')[-1] == 'csv':
        df = pd.read_csv(FILE_NAME)
        np_array = df.values
    else:
        np_array = np.loadtxt(FILE_NAME)

    if MAX_RECORDS != None: np_array = np_array[:MAX_RECORDS]
    data = coo_matrix((np_array[:, 2], \
        (np.array(np_array[:, 0], dtype=np.int32), np.array(np_array[:, 1], dtype=np.int32)))).toarray()
    n_train = int(data.shape[0] * TRAIN_RATIO)
    n_test = data.shape[0] - n_train
    error = 0
    good = 0
    n_total = 0
    rmse = 0
    for user in range(n_train, data.shape[0]):
        for item in range(0, data.shape[1]):
            if data[user][item] != 0:
                pred = predict_rating_user_cf(data[user], item, data[:n_train, :])
                real = data[user][item]
                error += (np.abs(pred - real))
                rmse += (pred - real)**2
                good += (pred >= THRESHOLD) == (real >= THRESHOLD)
                n_total += 1
    print("n_train: %d, n_test: %d, n_total: %d, error: %.3f, rmse: %.3f, accuracy: %.3f" \
        %(n_train, n_test, n_total, error/n_total, (rmse/n_total)**0.5, good/n_total))

def test_item_cf():
    if FILE_NAME.split('.')[-1] == 'csv':
        df = pd.read_csv(FILE_NAME)
        np_array = df.values
    else:
        np_array = np.loadtxt(FILE_NAME)

    if MAX_RECORDS != None: np_array = np_array[:MAX_RECORDS]
    data = coo_matrix((np_array[:, 2], \
        (np.array(np_array[:, 0], dtype=np.int32), np.array(np_array[:, 1], dtype=np.int32)))).toarray()
    n_train = int(data.shape[1] * TRAIN_RATIO)
    n_test = data.shape[1] - n_train
    n_true = 0
    n_total = 0
    for item in range(n_train, data.shape[1]):
        for user in range(0, data.shape[0]):
            if data[user][item] != 0:
                pred = predict_rating_item_cf_normalized(data[:, item], user, data[:, :n_train]) >= 3.5
                real = data[user][item] >= 3.5
                if pred == real: n_true += 1
                n_total += 1
    print("n_train: %d, n_test: %d, n_total: %d, accuracy: %.3f"%(n_train, n_test, n_total, n_true/n_total))

if __name__ == '__main__':
    test_user_cf()
