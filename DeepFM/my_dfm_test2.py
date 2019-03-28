import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from time import time
from sklearn.metrics import roc_auc_score
from metrics import gini_norm


class my_DeepFM(object):
    def __init__(self,feature_size,field_size,
                 embedding_size=8,dropout_fm=[1.0,1.0],
                 deep_layers=[32,32],dropout_deep=[0.5,0.5,0.5],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10,batch_size=256,
                 learning_rate=0.01,optimizer='adam',
                 batch_norm=0,batch_norm_decay=0.995,
                 verbose=False,random_seed=2018,
                 use_fm=True,use_deep=True,
                 loss_type='logloss',eval_metric=roc_auc_score,
                 l2_reg=0.0,greater_is_better=True):

        assert (use_fm or use_deep)
        assert loss_type in ['logloss','mse'],'loss_type can ben  for classification task or mse for regression task'

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layer_activation
        self.use_fm = use_fm
        self.use_deep =use_deep
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
        # self._init_graph()
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')  #####None*f
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')  # None*F
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
        self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
        self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_deep')
        # self.train_phase = tf.placeholder(tf.bool, name='train_phase')

        self._training = tf.placeholder_with_default(False, shape=(), name='training')
        # self._session = None

        self._init_graph()

    def _initialize_weight(self):

        weights = dict()
        weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size,self.embedding_size],0.0,0.01),name='feature_embeddings')###feature_size*k
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size,1],0.0,1),name='feature_bias')#####feature_size*1
        ###deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size*self.embedding_size
        glorot = np.sqrt(2/(input_size+self.deep_layers[0]))
        weights['layer_0'] = tf.Variable(
            tf.random_normal([input_size,self.deep_layers[0]],0.0,glorot),name='layer_0',dtype=tf.float32
        )
        weights['bias_0'] = tf.Variable(tf.random_normal([1,self.deep_layers[0]],0.0,glorot),name='bias_0',dtype=tf.float32)

        for i in range(1,num_layer):
            glorot = 2/(self.deep_layers[i-1]+self.deep_layers[i])
            weights['layer_'+str(i)] = tf.Variable(
                tf.random_normal([self.deep_layers[i-1], self.deep_layers[i]], 0.0, glorot), name='layer_'+str(i), dtype=tf.float32
            )
            weights['bias_'+str(i)] = tf.Variable(tf.random_normal([1,self.deep_layers[i]],0.0,glorot),name='bias_'+str(i),dtype=tf.float32)

        #########concat
        if self.use_fm and self.use_deep:
            input_size = self.field_size+self.embedding_size+self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size+self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0/(input_size+1))
        weights['concat_projection'] = tf.Variable(
            tf.random_normal([input_size,1],0.0,glorot,name='concat_projection',dtype=tf.float32)
        )
        weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=tf.float32)

        return weights

    def _init_graph(self):

        tf.set_random_seed(self.random_seed)
        self.weights = self._initialize_weight()
        ####
        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'],self.feat_index)  #n*f*k
        feat_value = tf.reshape(self.feat_value,shape=[-1,self.field_size,1])
        self.embeddings = tf.multiply(self.embeddings,feat_value)
        ##----first order term
        self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'],self.feat_index)#n*f*1
        self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order,feat_value),axis=2)##n*f
        self.y_first_order = tf.nn.dropout(self.y_first_order,self.dropout_keep_fm[0])
        #######second order-------
        ####sum square part
        self.summed_features_emb = tf.reduce_sum(self.embeddings,1)###n*k
        self.summed_features_emb_square = tf.square(self.summed_features_emb)##n*k
        #####squzre sum
        self.square_features_emb = tf.square(self.embeddings)
        self.square_sum_features_emb = tf.reduce_sum(self.square_features_emb,1)####n*k
        self.y_second_order = 0.5*tf.subtract(self.summed_features_emb_square,self.square_sum_features_emb)##n*k
        self.y_second_order = tf.nn.dropout(self.y_second_order,self.dropout_keep_fm[1])

        #######----------deep layers
        self.y_deep = tf.reshape(self.embeddings,shape=[-1,self.field_size*self.embedding_size])
        self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[0])
        for i in range(0,len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights['layer_'+str(i)]),self.weights['bias_'+str(i)])
            if  self.batch_norm==1 and self._training == True:
                self.y_deep = tf.layers.batch_normalization(self.y_deep,training=self._training,momentum=self.batch_norm_decay)
                print('user batch norm')
                # self.y_deep = tf.contrib.layers.batch_norm(self.y_deep,decay=self.batch_norm_decay,center=True,is_training=True,scope='bn_'+str(i))

            self.y_deep = self.deep_layers_activation(self.y_deep)
            self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[1+i])

        #######-------deepfm-------
        if self.use_fm and self.use_deep:
            concat_input = tf.concat([self.y_first_order,self.y_second_order,self.y_deep],axis=1)
        elif self.use_fm:
            concat_input = tf.concat([self.y_first_order,self.y_second_order],axis=1)
        elif self.use_deep:
            concat_input = self.y_deep

        self.out = tf.add(tf.matmul(concat_input,self.weights['concat_projection']),self.weights['concat_bias'])

        ##l0ss
        if self.loss_type == 'logloss':
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label,self.out)
        elif self.loss_type == 'mse':
            self.loss = tf.nn.l2_loss(tf.subtract(self.label,self.out))
        if self.l2_reg>0:
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['concat_projection'])
            if self.use_deep:
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights['layer_'+str(i)])

        ######optimizer

        if self.optimizer_type =='adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8)
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


        ######number of params
        total_paramters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()
            variable_paramters = 1
            for dim in shape:
                variable_paramters *=dim.value
            total_paramters +=variable_paramters
        if self.verbose >0:
            print('params:%d'%total_paramters)


