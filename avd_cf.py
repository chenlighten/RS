import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

TRAIN_RATIO = 0.9
K = 10
MAX_RECORDS = 30000
THRESHOLD = 3.5
GAMMA = 25
FILE_NAME = './data/ml-latest-small/ratings.csv'
# FILE_NAME = './data/demo_data'
# FILE_NAME = './data/ml_small_processed'

LAMBDA_1 = 0.01
LAMBDA_2 = 25
LAMBDA_3 = 10
LAMBDA_8 = 100

class sys():
    def __init__():
        if FILE_NAME.split('.')[-1] == 'csv':
            ratings = pd.read_csv(FILE_NAME)
            ratings = ratings.values
        else:
            ratings = np.loadtxt(FILE_NAME)

        users = []
        items = []
        for i in range(len(ratings)):
            users.append(ratings[i][0])
            items.append(ratings[i][1])
        users = set(users)
        items = set(items)
        self.user_num = len(users)
        self.item_num = len(items)
        
        self.avg_rating = np.mean(ratings[:, 2])

        self.ratings = coo_matrix((ratings[:, 2], (ratings[:, 0], ratings[:, 1]))).toarray()

    def pre_process(self):
        self.b_u = np.zeros([self.user_num])
        self.b_i = np.zeros([self.item_num])
        for i in range(self.item_num):
            self.b_i[i] = np.sum(self.ratings[self.ratings[:, i] != 0, i] - self.avg_rating) / \
                (LAMBDA_2 + len(self.ratings[self.ratings[:, i] != 0]))
        for u in range(self.user_num):
            index = self.ratings[u] != 0
            self.b_u[u] = np.sum(self.ratings[index] - self.avg_rating - self.b_i[index]) / \
                (LAMBDA_3 + len(self.ratings[index]))
        self.b = self.b_i + self.b_u.reshape([-1, 1]) + self.avg_rating
        
        self.sim = np.zeros([self.item_num, self.item_num])
        for i in range(self.item_num):
            for j in range(self.item_num):
                self.sim[i][j] = self.sim[j][i] = pearson_sim(i, j)

    def pearson_sim(self, i, j):
        index = (self.ratings[:, i] != 0) * (self.ratings[:, j] != 0)
        p =  np.mean(np.dot(self.ratings[index, i] - self.b[index, i]), (self.ratings[index, j] - self.b[index, j])) / \
            (np.sum((self.ratings[index, i] - self.b[index, i])**2) * np.sum((self.ratings[index, j] - self.b[index, j])**2))**0.5
        n = len(self.ratings[index])
        return (n - 1) / (n - 1 + LAMBDA_8) * p

    


    
    