import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

class my_DCN(BaseEstimator, TransformerMixin):

    def __init__(self, cate_feature_size, field_size,numeric_feature_size,
                 embedding_size=8,
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True,cross_layer_num=3):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.cate_feature_size = cate_feature_size
        self.numeric_feature_size = numeric_feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.total_size = self.field_size * self.embedding_size + self.numeric_feature_size
        self.deep_layers = deep_layers
        self.cross_layer_num = cross_layer_num
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.l2_reg = l2_reg

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

#        self._init_graph()

    def _init_graph(self):
        # self.graph = tf.Graph()

        tf.set_random_seed(self.random_seed)

        # self.weights = self._initialize_weights()

        # model
        self.weights = dict()
        #######embedding  layder
        # embeddings

        self.weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.cate_feature_size,self.embedding_size],0.0,0.01),name='feature_embeddings')
        self.weights['feature_bias'] = tf.Variable(tf.random_normal([self.cate_feature_size,1],0.0,0.01),name='feature_bias')
        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index) # N * F * K
        feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])#####N*F*1
        self.embeddings = tf.multiply(self.embeddings,feat_value)
        ############embedding后的特征和数值类特征进行合并
        self.x0 = tf.concat([self.numeric_value,
                             tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])]
                            ,axis=1)


        # deep part

        self.y_deep = tf.nn.dropout(self.x0,self.dropout_keep_deep[0])

        for i in range(0,len(self.deep_layers)):
            if i==0:
                glorot = np.sqrt(2.0 / (self.total_size + self.deep_layers[0]))
                self.weights['deep_layer_0'] = tf.Variable(tf.random_normal([self.total_size,self.deep_layers[0]],0,glorot),name='deep_layer_0')

                self.weights['deep_bias_0'] = tf.Variable(tf.random_normal([1,self.deep_layers[0]],0,glorot),name='deep_bias_0')

            else:
                glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
                self.weights['deep_layer_'+str(i)] = tf.Variable(tf.random_normal([self.deep_layers[i-1],self.deep_layers[i]],0,glorot),name='deep_layer_'+str(i))
                self.weights['deep_bias_'+str(i)] = tf.Variable(tf.random_normal([1,self.deep_layers[i]],0,glorot),name='deep_bias_'+str(i))
            self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights["deep_layer_%d" %i]), self.weights["deep_bias_%d"%i])
            if self.batch_norm==1 and self.train_phase==True:
                self.y_deep = tf.layers.batch_normalization(self.y_deep,momentum=self.batch_norm_decay,training=self.train_phase)
            self.y_deep = self.deep_layers_activation(self.y_deep)
            self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[i+1])
        ########## # cross_part
        # self._x0 = tf.reshape(self.x0, (-1, self.total_size, 1))
        # x_l = self._x0
        # for l in range(self.cross_layer_num):
        #     x_l = tf.tensordot(tf.matmul(self._x0, x_l, transpose_b=True),
        #                         self.weights["cross_layer_%d" % l],1) + self.weights["cross_bias_%d" % l] + x_l
        #
        # self.cross_network_out = tf.reshape(x_l, (-1, self.total_size))

        ##############my cross net

        x_l = self.x0
        for i in range(self.cross_layer_num):
            self.weights['cross_layer_%d'%i] = tf.Variable(tf.random_normal([self.total_size,1],0,glorot),name='cross_layer_'+str(i))
            self.weights['cross_bias_%d'%i] = tf.Variable(tf.random_normal([self.total_size,1],0,glorot),name='cross_bias_'+str(i))
            xlw = tf.matmul(x_l,self.weights['cross_layer_%d'%i])
            x_l = self.x0*xlw+x_l+tf.reshape(self.weights["cross_bias_%d" %i],(-1,self.total_size))
        self.cross_network_out = x_l

        # concat_part
        concat_input = tf.concat([self.cross_network_out, self.y_deep], axis=1)
        ############z最后一层全连接
        input_size = self.total_size + self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        self.weights['concat_projection'] = tf.Variable(tf.random_normal([input_size,1],0,glorot),name='concat_projection')
        self.weights['concat_bias'] = tf.Variable(tf.constant(0.01),name='concat_bias')
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
            for i in range(len(self.deep_layers)):
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["deep_layer_%d" % i])
            for i in range(self.cross_layer_num):
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["cross_layer_%d" % i])


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



    def get_batch(self,Xi,Xv,Xv2,y,batch_size,index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end],Xv[start:end],Xv2[start:end],[y_ for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c,d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)

    def predict(self, Xi, Xv,Xv2,y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y

        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.numeric_value: Xv2,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                     self.train_phase:False}

        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss


    def fit(self, cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train,
            cate_Xi_valid=None, cate_Xv_valid=None, numeric_Xv_valid=None,y_valid=None,
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
        
        


        print(len(cate_Xi_train))
        print(len(cate_Xv_train))
        print(len(numeric_Xv_train))
        print(len(y_train))
        has_valid = cate_Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(cate_Xi_train, cate_Xv_train,numeric_Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            
            for i in range(total_batch):
                cate_Xi_batch, cate_Xv_batch,numeric_Xv_batch, y_batch = self.get_batch(cate_Xi_train, cate_Xv_train, numeric_Xv_train,y_train, self.batch_size, i)

                feed_dict = {self.feat_index:cate_Xi_batch,
                     self.feat_value:cate_Xv_batch,
                     self.numeric_value:numeric_Xv_batch,
                     self.label:np.array(y_batch).reshape(-1,1),
                     self.dropout_keep_deep:self.dropout_dep,
                     self.train_phase:True}

                loss,opt = self.sess.run([self.loss,self.train_step],feed_dict=feed_dict)
#                
                
#                self.fit_on_batch(cate_Xi_batch, cate_Xv_batch,numeric_Xv_batch, y_batch)
            loss_tr = self.sess.run(self.loss,feed_dict = {
                    self.feat_index:cate_Xi_train,
                     self.feat_value:cate_Xv_train,
                     self.numeric_value:numeric_Xv_train,
                     self.label:np.array(y_train).reshape(-1,1),
                     self.dropout_keep_deep:[1.0] * len(self.dropout_dep),
                                        })
            
#            loss_tr = self.predict(cate_Xi_train,cate_Xv_train,numeric_Xv_train, y_train)
            self.train_result.append(loss_tr)


            if has_valid:
                
#                y_valid = np.array(y_valid).reshape((-1,1))
#                loss_va = self.predict(cate_Xi_valid, cate_Xv_valid, numeric_Xv_valid, y_valid)
                
                loss_va = self.sess.run(
                        self.loss,feed_dict = {
                    self.feat_index:cate_Xi_valid,
                     self.feat_value:cate_Xv_valid,
                     self.numeric_value:numeric_Xv_valid,
                     self.label:np.array(y_valid).reshape(-1,1),
                     self.dropout_keep_deep:[1.0] * len(self.dropout_dep),
                       } )
                
                self.valid_result.append(loss_va)
#                
#                print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
#                          % (epoch + 1, loss_tr, loss_va, time() - t1))
                print("epoch",(epoch+1),'loss_train %.4f'%(loss_tr),"loss_valid %.4f"%(loss_va),'time [%0.1f]'%(time()-t1))













