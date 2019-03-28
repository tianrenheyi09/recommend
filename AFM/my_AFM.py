import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

class my_AFM(BaseEstimator, TransformerMixin):

    def __init__(self, feature_size, field_size,
                 embedding_size=8,attention_size=10,
                 deep_layers=[32, 32], deep_init_size=50,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True,use_inner=True):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.cate_feature_size = feature_size
        self.feature_size = feature_size

        self.field_size = field_size
        self.embedding_size = embedding_size
        self.attention_size = attention_size


        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layers_activation


        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

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



    def _init_graph(self):


        tf.set_random_seed(self.random_seed)

        # model
        self.weights = dict()
        #######embedding  layder
        # embeddings

        self.weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size,self.embedding_size],0.0,0.01),name='feature_embeddings')
        self.weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size,1],0.0,0.01),name='feature_bias')
        self.weights['bias'] = tf.Variable(tf.constant(0.1),name='bias')

        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
        feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])#####N*F*1
        self.embeddings = tf.multiply(self.embeddings,feat_value)#N*F*K

        # attention part
        glorot = np.sqrt(2.0 / (self.attention_size + self.embedding_size))

        self.weights['attention_w'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.attention_size)),
            dtype=tf.float32, name='attention_w')

        self.weights['attention_b'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.attention_size,)),
                                             dtype=tf.float32, name='attention_b')

        self.weights['attention_h'] = tf.Variable(np.random.normal(loc=0, scale=1, size=(self.attention_size,)),
                                             dtype=tf.float32, name='attention_h')

        self.weights['attention_p'] = tf.Variable(np.ones((self.embedding_size, 1)), dtype=np.float32)

        ####element_wise
        element_wise_product_list = []
        for i in range(self.field_size):
            for j in range(i + 1, self.field_size):
                element_wise_product_list.append(
                    tf.multiply(self.embeddings[:, i, :], self.embeddings[:, j, :]))  # None * K

        self.element_wise_product = tf.stack(element_wise_product_list)  # (F * F - 1 / 2) * None * K
        self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2],
                                                 name='element_wise_product')  # None * (F * F - 1 / 2) *  K

        # self.interaction

        # attention part
        num_interactions = int(self.field_size * (self.field_size - 1) / 2)
        # wx+b -> relu(wx+b) -> h*relu(wx+b)
        self.attention_wx_plus_b = tf.reshape(
            tf.add(tf.matmul(tf.reshape(self.element_wise_product, shape=(-1, self.embedding_size)),
                             self.weights['attention_w']),
                   self.weights['attention_b']),
            shape=[-1, num_interactions, self.attention_size])  # N * ( F * F - 1 / 2) * A

        self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(tf.nn.relu(self.attention_wx_plus_b),
                                                              self.weights['attention_h']),
                                                  axis=2, keep_dims=True))  # N * ( F * F - 1 / 2) * 1

        self.attention_exp_sum = tf.reduce_sum(self.attention_exp, axis=1, keep_dims=True)  # N * 1 * 1

        self.attention_out = tf.div(self.attention_exp, self.attention_exp_sum,
                                    name='attention_out')  # N * ( F * F - 1 / 2) * 1

        self.attention_x_product = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product), axis=1,
                                                 name='afm')  # N * K

        self.attention_part_sum = tf.matmul(self.attention_x_product, self.weights['attention_p'])  # N * 1

        # first order term
        self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
        self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)

        # bias
        self.y_bias = self.weights['bias'] * tf.ones_like(self.label)

        # out
        self.out = tf.add_n([tf.reduce_sum(self.y_first_order, axis=1, keep_dims=True),
                             self.attention_part_sum,
                             self.y_bias], name='out_afm')




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



        # number of params
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


    def fit(self, Xi_train, Xv_train,y_train,
            Xi_valid=None, Xv_valid=None,y_valid=None,
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

        #init
        self._init_graph()
        # self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        


        print(len(Xi_train))
        print(len(Xv_train))
        print(len(y_train))
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train,Xv_train,y_train)
            total_batch = int(len(y_train) / self.batch_size)
            
            for i in range(total_batch):
                Xi_batch,Xv_batch,y_batch = self.get_batch(Xi_train,Xv_train,y_train, self.batch_size, i)

                feed_dict = {self.feat_index:Xi_batch,
                     self.feat_value:Xv_batch,
                     self.label:np.array(y_batch).reshape(-1,1),
                     self.dropout_keep_deep:self.dropout_dep,
                     self.train_phase:True}

                loss,opt = self.sess.run([self.loss,self.train_step],feed_dict=feed_dict)
#
            loss_tr = self.sess.run(self.loss,feed_dict = {
                    self.feat_index:Xi_train,
                     self.feat_value:Xv_train,
                     self.label:np.array(y_train).reshape(-1,1),
                     self.dropout_keep_deep:[1.0] * len(self.dropout_dep),
                                        })

            self.train_result.append(loss_tr)


            if has_valid:

                loss_va = self.sess.run(
                        self.loss,feed_dict = {
                    self.feat_index:Xi_valid,
                     self.feat_value:Xv_valid,
                     self.label:np.array(y_valid).reshape(-1,1),
                     self.dropout_keep_deep:[1.0] * len(self.dropout_dep),
                       } )
                
                self.valid_result.append(loss_va)

                print("epoch",(epoch+1),'loss_train %.4f'%(loss_tr),"loss_valid %.4f"%(loss_va),'time [%0.1f]'%(time()-t1))













