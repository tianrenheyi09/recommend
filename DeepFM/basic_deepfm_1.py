import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from metrics import gini_norm

class DeepFM_one(BaseEstimator, TransformerMixin):

    def __init__(self, feature_size, field_size,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layer_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

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

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32,
                                             shape=[None,None],
                                             name='feat_index')
            self.feat_value = tf.placeholder(tf.float32,
                                           shape=[None,None],
                                           name='feat_value')

            self.label = tf.placeholder(tf.float32,shape=[None,1],name='label')
            self.dropout_keep_fm = tf.placeholder(tf.float32,shape=[None],name='dropout_keep_fm')
            self.dropout_keep_deep = tf.placeholder(tf.float32,shape=[None],name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool,name='train_phase')

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
            feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
            self.embeddings = tf.multiply(self.embeddings,feat_value)


            # first order term
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'],self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order,feat_value),2)
            self.y_first_order = tf.nn.dropout(self.y_first_order,self.dropout_keep_fm[0])

            # second order term
            # sum-square-part
            self.summed_features_emb = tf.reduce_sum(self.embeddings,1) # None * k
            self.summed_features_emb_square = tf.square(self.summed_features_emb) # None * K

            # squre-sum-part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            #second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,self.squared_sum_features_emb)
            self.y_second_order = tf.nn.dropout(self.y_second_order,self.dropout_keep_fm[1])


            # Deep component
            self.y_deep = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[0])

            for i in range(0,len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["layer_%d" %i]), self.weights["bias_%d"%i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])


            #----DeepFM---------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep

            self.out = tf.add(tf.matmul(concat_input,self.weights['concat_projection']),self.weights['concat_bias'])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])


            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)


            #init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        weights = dict()

        #embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size,self.embedding_size],0.0,0.01),
            name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size,1],0.0,1.0),name='feature_bias')


        #deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0/(input_size + self.deep_layers[0]))

        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(input_size,self.deep_layers[0])),dtype=np.float32
        )
        weights['bias_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(1,self.deep_layers[0])),dtype=np.float32
        )


        for i in range(1,num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]


        # final concat projection layer

        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0/(input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32)


        return weights


    def get_batch(self,Xi,Xv,y,batch_size,index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end],Xv[start:end],[y_ for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv)
        return self.eval_metric(y, y_pred)

    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: np.array(Xi_batch),
                         self.feat_value: np.array(Xv_batch),
                         self.label: np.array(y_batch).reshape((-1,1)),
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred


    def fit_on_batch(self,Xi,Xv,y):
        feed_dict = {self.feat_index:Xi,
                     self.feat_value:Xv,
                     self.label:y,
                     self.dropout_keep_fm:self.dropout_fm,
                     self.dropout_keep_deep:self.dropout_dep,
                     self.train_phase:True}

        loss,opt = self.sess.run([self.loss,self.optimizer],feed_dict=feed_dict)

        return loss
    def my_fit(self,Xi_train_, Xv_train_, y_train_,
            Xi_valid_=None, Xv_valid_=None, y_valid_=None):

        max_checks_without_progress = 10
        checks_without_progress = 0
        best_gini = 0

        loss_train = 0
        loss_test = 0
        self.gini_train = []
        self.gini_valid = []
        for epoch in range(self.epoch):
            t1 = time()

            pre_train = []
            pre_valid = []

            self.shuffle_in_unison_scary(Xi_train_, Xv_train_, y_train_)
            total_batch = int((len(y_train_) - 1) / self.batch_size) + 1
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train_, Xv_train_, y_train_, self.batch_size, i)
                feed_dict = {self.feat_index: np.array(Xi_batch),
                             self.feat_value: np.array(Xv_batch),
                             self.label: np.array(y_batch).reshape((-1, 1)),
                             self.dropout_keep_fm: self.dropout_fm,
                             self.dropout_keep_deep: self.dropout_dep,
                             self.train_phase: True
                             }

                loss, opt, train_out = self.sess.run((self.loss, self.optimizer, self.out), feed_dict=feed_dict)
                # if extra_update_ops:
                #     sess.run(extra_update_ops,feed_dict=feed_dict)
                loss_train += loss
                pre_train.append(train_out)
                # dfm.fit_on_batch(Xi_batch, Xv_batch, y_batch)
            loss_train /= total_batch
            pre_train = [y for x in pre_train for y in x]
            sig_gini_train = gini_norm(y_train_, pre_train)
            self.gini_train.append(sig_gini_train)

            feed_dict = {self.feat_index: np.array(Xi_valid_),
                         self.feat_value: np.array(Xv_valid_),
                         self.label: np.array(y_valid_).reshape((-1, 1)),
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                         self.train_phase: False

                         }
            loss_test, valid_out = self.sess.run((self.loss, self.out), feed_dict=feed_dict)
            pre_valid.append(valid_out)
            pre_valid = [y for x in pre_valid for y in x]
            sig_gini_valid = gini_norm(y_valid_, pre_valid)
            self.gini_valid.append(sig_gini_valid)

            if sig_gini_valid > best_gini:
                gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

                best_params = {gvar.op.name: value for gvar, value in zip(gvars, self.sess.run(gvars))}
                best_gini = sig_gini_valid
                checks_without_progress = 0
            else:
                checks_without_progress += 1

            print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                  % (epoch + 1, sig_gini_train, sig_gini_valid, time() - t1))

            if checks_without_progress > max_checks_without_progress:
                print('early stopping!')
                break
        ##########将训练过程中保存的最好的参数重新返回到模型参数，此时得到的是最好的模型
        if best_params:
            gvars_names = list(best_params.keys())
            assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name
                          in
                          gvars_names}
            init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
            feed_dict = {init_values[gvar_name]: best_params[gvar_name] for gvar_name in gvars_names}
            self.sess.run(assign_ops, feed_dict=feed_dict)

        return self
    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):

        has_valid = Xv_valid is not None
        self.gini_train = []
        self.gini_valid = []

        for epoch in range(self.epoch):
            pre_train = []
            pre_valid = []
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int((len(y_train) - 1) / self.batch_size) + 1
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)

                feed_dict = {self.feat_index: np.array(Xi_batch),
                             self.feat_value: np.array(Xv_batch),
                             self.label: np.array(y_batch).reshape((-1, 1)),
                             self.dropout_keep_fm: self.dropout_fm,
                             self.dropout_keep_deep: self.dropout_dep,
                             self.train_phase: True
                             }

                # loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                loss, opt, train_out = self.sess.run((self.loss, self.optimizer, self.out), feed_dict=feed_dict)
                # pre_train.append(train_out)
            # dfm.fit_on_batch(Xi_batch, Xv_batch, y_batch)

            for i in range(total_batch):
                dummy = [1] * len(Xi_train)
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, dummy, self.batch_size, i)
                num_batch = len(y_batch)
                feed_dict = {self.feat_index: np.array(Xi_batch),
                             self.feat_value: np.array(Xv_batch),
                             self.label: np.array(y_batch).reshape((-1, 1)),
                             self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                             self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                             self.train_phase: False

                             }
                loss, train_out = self.sess.run((self.loss, self.out), feed_dict=feed_dict)
                if i == 0:
                    pre_train = np.reshape(train_out, (num_batch,))
                else:
                    pre_train = np.concatenate((pre_train, np.reshape(train_out, (num_batch,))))

            sig_gini_train = gini_norm(y_train,pre_train)
            # sig_gini_train = self.evaluate(Xi_train, Xv_train, y_train)
            self.gini_train.append(sig_gini_train)


            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result)
                feed_dict = {self.feat_index: np.array(Xi_valid),
                             self.feat_value: np.array(Xv_valid),
                             self.label: np.array(y_valid).reshape((-1, 1)),
                             self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                             self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                             self.train_phase: False

                             }
                loss_test, valid_out = self.sess.run((self.loss, self.out), feed_dict=feed_dict)
                pre_valid.append(valid_out)
                pre_valid = [y for x in pre_valid for y in x]
                sig_gini_valid = gini_norm(y_valid, pre_valid)
                self.gini_valid.append(sig_gini_valid)


            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f,my-train=%.4f, my_valid=%.4f [%.1f s],"
                        % (epoch + 1, train_result, valid_result,sig_gini_train,sig_gini_valid, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

    def my_predict_prob(self,Xi_valid_,Xv_valid_):
        ##########valid  预测
        dummy_y = [1] * len(Xi_valid_)
        feed_dict = {self.feat_index: np.array(Xi_valid_),
                     self.feat_value: np.array(Xv_valid_),
                     self.label: np.array(dummy_y).reshape((-1, 1)),
                     self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep)
                     # dfm._training:False
                     }
        _, valid_out = self.sess.run((self.loss, self.out), feed_dict=feed_dict)

        return valid_out

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False













