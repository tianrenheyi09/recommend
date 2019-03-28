import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

class my_NFM(BaseEstimator, TransformerMixin):

    def __init__(self, feature_size, field_size,
                 embedding_size=8,
                 deep_layers=[32, 32], deep_init_size = 50,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                greater_is_better=True,
                 use_inner=True):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layer_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result,self.valid_result = [],[]

        self.use_inner = use_inner

        ################输入变量
        self.feat_index = tf.placeholder(tf.int32,
                                         shape=[None, None],
                                         name='feat_index')
        self.feat_value = tf.placeholder(tf.float32,
                                         shape=[None, None],
                                         name='feat_value')

        self.numeric_value = tf.placeholder(tf.float32, [None, None], name='num_value')

        self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
        self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_deep_deep')
        self.train_phase = tf.placeholder_with_default(False, shape=(), name='train_phase')

        # self._init_graph()

    def _init_graph(self):

        tf.set_random_seed(self.random_seed)

        self.feat_index = tf.placeholder(tf.int32,
                                         shape=[None,None],
                                         name='feat_index')
        self.feat_value = tf.placeholder(tf.float32,
                                       shape=[None,None],
                                       name='feat_value')

        self.label = tf.placeholder(tf.float32,shape=[None,1],name='label')
        self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')
        self.train_phase = tf.placeholder(tf.bool,name='train_phase')

        self.weights = {}

        # Embeddings
        self.weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
        self.weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 0.01),
                                                   name='feature_bias')
        self.weights['bias'] = tf.Variable(tf.constant(0.1), name='bias')

        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
        feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
        self.embeddings = tf.multiply(self.embeddings,feat_value) # N * F * K

        # first order term
        self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
        self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)


        # second order term
        # sum-square-part
        self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * k
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        # squre-sum-part
        self.squared_features_emb = tf.square(self.embeddings)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        # second order
        self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)

        # Deep component
        self.y_deep = self.y_second_order

        num_layer = len(self.deep_layers)
        input_size = self.embedding_size


        for i in range(0, len(self.deep_layers)):
            if i == 0:
                glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
                self.weights['deep_layer_0'] = tf.Variable(
                    tf.random_normal([input_size, self.deep_layers[0]], 0, glorot), name='deep_layer_0')

                self.weights['deep_bias_0'] = tf.Variable(tf.random_normal([1, self.deep_layers[0]], 0, glorot),
                                                          name='deep_bias_0')

            else:
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
                self.weights['deep_layer_' + str(i)] = tf.Variable(
                    tf.random_normal([self.deep_layers[i - 1], self.deep_layers[i]], 0, glorot),
                    name='deep_layer_' + str(i))
                self.weights['deep_bias_' + str(i)] = tf.Variable(
                    tf.random_normal([1, self.deep_layers[i]], 0, glorot), name='deep_bias_' + str(i))

            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["deep_layer_%d" % i]),
                                 self.weights["deep_bias_%d" % i])
            if self.batch_norm == 1 and self.train_phase == True:
                self.y_deep = tf.layers.batch_normalization(self.y_deep, momentum=self.batch_norm_decay,
                                                            training=self.train_phase)
            self.y_deep = self.deep_layers_activation(self.y_deep)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])


            # self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
            # self.y_deep = self.deep_layers_activation(self.y_deep)
            # self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])

        # bias
        self.y_bias = self.weights['bias'] * tf.ones_like(self.label)

        # out
        self.out = tf.add_n([tf.reduce_sum(self.y_first_order,axis=1,keep_dims=True),
                             tf.reduce_sum(self.y_deep,axis=1,keep_dims=True),
                             self.y_bias])

        # loss
        if self.loss_type == "logloss":
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)
        elif self.loss_type == "mse":
            self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

        if self.optimizer_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                self.train_step = self.optimizer.minimize(self.loss)
        elif self.optimizer_type == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                self.train_step = self.optimizer.minimize(self.loss)
        elif self.optimizer_type == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                self.train_step = self.optimizer.minimize(self.loss)
        elif self.optimizer_type == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                self.train_step = self.optimizer.minimize(self.loss)

            # #init
            # self.saver = tf.train.Saver()
            # init = tf.global_variables_initializer()
            # self.sess = tf.Session()
            # self.sess.run(init)
            #
            # # number of params
            # total_parameters = 0
            # for variable in self.weights.values():
            #     shape = variable.get_shape()
            #     variable_parameters = 1
            #     for dim in shape:
            #         variable_parameters *= dim.value
            #     total_parameters += variable_parameters
            # if self.verbose > 0:
            #     print("#params: %d" % total_parameters)




    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def predict(self, Xi, Xv,y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                     self.train_phase:False}

        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss

    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """

        # init
        self._init_graph()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        print(len(Xi_train))
        print(len(Xv_train))
        print(len(y_train))
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)

            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)

                feed_dict = {self.feat_index: Xi_batch,
                             self.feat_value: Xv_batch,
                             self.label: np.array(y_batch).reshape(-1, 1),
                             self.dropout_keep_deep: self.dropout_dep,
                             self.train_phase: True}

                loss, opt = self.sess.run([self.loss, self.train_step], feed_dict=feed_dict)
                #
            loss_tr = self.sess.run(self.loss, feed_dict={
                self.feat_index: Xi_train,
                self.feat_value: Xv_train,
                self.label: np.array(y_train).reshape(-1, 1),
                self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
            })

            self.train_result.append(loss_tr)

            if has_valid:
                loss_va = self.sess.run(
                    self.loss, feed_dict={
                        self.feat_index: Xi_valid,
                        self.feat_value: Xv_valid,
                        self.label: np.array(y_valid).reshape(-1, 1),
                        self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                    })

                self.valid_result.append(loss_va)

                print("epoch", (epoch + 1), 'loss_train %.4f' % (loss_tr), "loss_valid %.4f" % (loss_va),
                      'time [%0.1f]' % (time() - t1))




