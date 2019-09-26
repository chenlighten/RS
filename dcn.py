import keras
from keras import layers
import tensorflow as tf
import numpy as np

class DCN:
    def __init__(self, dense_feat_dim, sparse_feat_dim, sparse_feat_num, emb_dim,
                 deep_layers=[32, 32, 32], cross_layers_num=3,
                 dropout=True, drop_keep_prob=[0.8, 0.8, 0.8],
                 learning_rate=0.01, l2_factor=0.01,
                 batch_size=64, epochs=100):
        self.dense_feat_dim = dense_feat_dim
        self.sparse_feat_dim = sparse_feat_dim
        self.sparse_feat_num = sparse_feat_num
        self.emb_dim = emb_dim
        self.deep_layers = deep_layers
        self.cross_layers_num = cross_layers_num
        self.dropout = dropout
        self.drop_keep_prob = drop_keep_prob
        self.learning_rate = learning_rate
        self.l2_factor = l2_factor
        self.batch_size = batch_size
        self.epochs = epochs

        self.feat_dim = self.dense_feat_dim + self.sparse_feat_num*self.emb_dim

        self._build_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    
    def _build_graph(self):
        # ----------------------- input placeholders ------------------------------
        self.dense_feat_in = tf.placeholder(tf.float32, (None, self.dense_feat_dim))
        self.sparse_feat_index_in = tf.placeholder(tf.int32, (None, self.sparse_feat_num))
        
        # ----------------------- embeddings ----------------------------
        self.embeddings = tf.Variable(np.random.normal(scale=0.1, size=(self.sparse_feat_dim, self.emb_dim)))
        self.embeded = tf.nn.embedding_lookup(self.embeddings, self.sparse_feat_index_in)
        self.embeded = tf.reshape(self.embeded, (-1, self.emb_dim*self.sparse_feat_num))
        
        self.origin_feat = tf.concat([self.dense_feat_in, self.embeded], axis=1)

        # ----------------------------- deep layers ----------------------------------
        hidden = self._build_deep_layer(self.origin_feat, self.feat_dim, self.deep_layers[0])
        for i in range(len(self.deep_layers) - 1):
            hidden = self._build_deep_layer(hidden, self.deep_layers[i], self.deep_layers[i + 1])
        self.deep_output = hidden

        # -------------------------------- cross layers --------------------------------
        for i in range(self.cross_layers_num):
            cross = self._build_cross_layer(self.origin_feat)
        self.cross_output = cross

        # ------------------------------ final output --------------------------------
        combined_output = tf.concat([self.cross_output, self.deep_output])
        weight = tf.Variable(np.random.normal(scale=0.1, size=(self.feat_dim + self.deep_layers[-1])))
        bias = tf.Variable(0)
        self.final_output = tf.nn.sigmoid(tf.matmul(combined_output, tf.transpose(weight)) + bias)

        # ----------------------------- loss and optimizer ---------------------------
        self.ground_truth = tf.placeholder(tf.float32, (None))
        self.target_loss = tf.nn.l2_loss(self.final_output - self.ground_truth)
        self.normal_loss = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss = self.target_loss + self.l2_factor * self.normal_loss

        self.optmizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)


    def _build_deep_layer(self, in_vec, in_units, out_units):
        glorot = np.sqrt(12/(in_units + out_units))
        weight = tf.Variable(np.random.normal(scale=glorot, size=(in_units, out_units)))
        bias = tf.Variable(np.zeros(shape=(out_units)))
        return tf.matmul(in_vec, weight) + bias

    
    def _build_cross_layer(self, in_vec):
        weight = tf.Variable(np.random.normal(scale=0.1, size=(self.feat_dim)))
        bias = tf.Variable(np.zeros(size=(self.feat_dim)))
        feature_crossing = tf.matmul(tf.matmul(tf.transpose(self.origin_feat), in_vec), weight).reshape((-1))
        return feature_crossing + bias + in_vec
    

    def fit(self, dense_feat, sparse_feat_index, ground_truth):
        random_state = np.random.get_state()
        dense_feat = np.random.shuffle(dense_feat)

        np.random.set_state(random_state)
        sparse_feat_index = np.random.shuffle(sparse_feat_index)

        np.random.set_state(random_state)
        ground_truth = np.random.shuffle(ground_truth)
        
        for i in range(0, len(dense_feat), self.batch_size):
            bound = min(len(dense_feat), i + self.batch_size)
            feed_dict = {
                self.dense_feat_in: dense_feat[i:bound],
                self.sparse_feat_index_in: sparse_feat_index[i:bound],
                self.ground_truth: ground_truth[i:bound]
            }
            _, error = self.sess.run([self.optmizer, self.loss], feed_dict)
            
