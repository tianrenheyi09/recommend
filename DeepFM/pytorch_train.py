import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from DataReader import FeatureDictionary, DataParser
from matplotlib import pyplot as plt

import config
from metrics import gini_norm
# from my_deepfm import DeepFM
from basic_DeepFM import DeepFM
# from my_deepfm_array import DeepFM
from my_dfm_test2 import my_DeepFM

def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ['id','target']]
        #df['missing_feat'] = np.sum(df[df[cols]==-1].values,axis=1)
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ['id','target']]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['target'].values

    X_test = dfTest[cols].values
    ids_test = dfTest['id'].values

    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain,dfTest,X_train,y_train,X_test,ids_test,cat_features_indices

def get_batch( Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], [y_ for y_ in y[start:end]]

def shuffle_in_unison_scary( a, b, c):
    res = list(zip(a,b,c))
    np.random.shuffle(res)
    a,b,c = zip(*res)

# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()

##########得到index和value
fd = FeatureDictionary(dfTrain=dfTrain,
                       dfTest=dfTest,
                       numeric_cols=config.NUMERIC_COLS,
                       ignore_cols=config.IGNORE_COLS)
data_parser = DataParser(feat_dict=fd)
# Xi_train ：列的序号
# Xv_train ：列的对应的值
Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

print(dfTrain.dtypes)


# ############随机打乱划分训练集和验证集
np.random.seed(2018)
shuftle_index = np.random.permutation(len(Xi_train))
train_idx = shuftle_index[0:int(0.08*len(Xi_train))]
valid_idx = shuftle_index[int(0.08*len(Xi_train)):int(0.1*len(Xi_train))]
_get = lambda x,l:[x[i] for i in l]
Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)


class Config(object):
    """
    用来存储一些配置信息
    """
    def __init__(self):
        self.feature_dict = None
        self.feature_size = None
        self.field_size = None
        self.embedding_size = 8

        self.epochs = 20
        self.deep_layers_activation = tf.nn.relu

        self.loss = "logloss"
        self.l2_reg = 0.1
        self.learning_rate = 0.1


config.feature_size = fd.feat_dim
config.field_size = len(Xi_train[0])
# 模型参数
deep_layers = [32,32]
config.embedding_size = 8
config.deep_layers_activation = tf.nn.relu

# BUILD THE WHOLE MODEL
tf.set_random_seed(2018)
# init_weight
weights = dict()
# Sparse Features 到 Dense Embedding的全连接权重。[其实是Embedding]
weights['feature_embedding'] = tf.Variable(initial_value=tf.random_normal(shape=[config.feature_size, config.embedding_size],mean=0,stddev=0.1),
                                           name='feature_embedding',
                                           dtype=tf.float32)
# Sparse Featues 到 FM Layer中Addition Unit的全连接。 [其实是Embedding,嵌入后维度为1]
weights['feature_bias'] = tf.Variable(initial_value=tf.random_uniform(shape=[config.feature_size, 1],minval=0.0,maxval=1.0),
                                      name='feature_bias',
                                      dtype=tf.float32)
# Hidden Layer
num_layer = len(deep_layers)
input_size = config.field_size * config.embedding_size
glorot = np.sqrt(2.0 / (input_size + deep_layers[0])) # glorot_normal: stddev = sqrt(2/(fan_in + fan_out))
weights['layer_0'] = tf.Variable(initial_value=tf.random_normal(shape=[input_size, deep_layers[0]],mean=0,stddev=glorot),
                                 dtype=tf.float32)
weights['bias_0'] = tf.Variable(initial_value=tf.random_normal(shape=[1, deep_layers[0]],mean=0,stddev=glorot),
                                dtype=tf.float32)
for i in range(1, num_layer):
    glorot = np.sqrt(2.0 / (deep_layers[i - 1] + deep_layers[i]))
    # deep_layer[i-1] * deep_layer[i]
    weights['layer_%d' % i] = tf.Variable(initial_value=tf.random_normal(shape=[deep_layers[i - 1], deep_layers[i]],mean=0,stddev=glorot),
                                          dtype=tf.float32)
    # 1 * deep_layer[i]
    weights['bias_%d' % i] = tf.Variable(initial_value=tf.random_normal(shape=[1, deep_layers[i]],mean=0,stddev=glorot),
                                         dtype=tf.float32)
# Output Layer
deep_size = deep_layers[-1]
fm_size = config.field_size + config.embedding_size
input_size = fm_size + deep_size
glorot = np.sqrt(2.0 / (input_size + 1))
weights['concat_projection'] = tf.Variable(initial_value=tf.random_normal(shape=[input_size,1],mean=0,stddev=glorot),
                                           dtype=tf.float32)
weights['concat_bias'] = tf.Variable(tf.constant(value=0.01), dtype=tf.float32)


# build_network
# feat_index = tf.placeholder(dtype=tf.int32, shape=[None, config.field_size], name='feat_index') # [None, field_size]
# feat_value = tf.placeholder(dtype=tf.float32, shape=[None, None], name='feat_value') # [None, field_size]
# label = tf.placeholder(dtype=tf.float16, shape=[None,1], name='label')
config.batch_size = 1024
shuffle_in_unison_scary(Xi_train_, Xv_train_, y_train_)
total_batch = int((len(y_train_)-1) / config.batch_size)+1

Xi_batch, Xv_batch, y_batch = get_batch(Xi_train_, Xv_train_, y_train_, config.batch_size, 1)
feat_index = np.array(Xi_batch)
feat_value = np.array(Xv_batch)
label = np.array(y_batch).reshape((-1,1))

# dfm.feat_index: np.array(Xi_batch),
#                        dfm.feat_value: np.array(Xv_batch),
#                        dfm.label: np.array(y_batch).reshape((-1,1)),
#                        dfm.dropout_keep_fm:dfm.dropout_fm,
#                        dfm.dropout_keep_deep: dfm.dropout_deep,
#                        dfm._training:True

# Sparse Features -> Dense Embedding
import torch
embeddings_origin = tf.nn.embedding_lookup(weights['feature_embedding'], ids=feat_index) # [None, field_size, embedding_size]
torch_embeddings_origin = torch.nn.Embedding(config.feature_size, config.embedding_size)(torch.from_numpy(feat_index).type(torch.LongTensor))# [None, field_size, embedding_size]


feat_value_reshape = tf.reshape(tensor=feat_value, shape=[-1, config.field_size, 1]) # -1 * field_size * 1
torch_feat_value_reshape = torch.reshape(torch.from_numpy(feat_value),[-1,config.field_size, 1]).type(torch.FloatTensor)

# --------- 一维特征 -----------
y_first_order = tf.nn.embedding_lookup(weights['feature_bias'], ids=feat_index) # [None, field_size, 1]
w_mul_x = tf.multiply(tf.cast(y_first_order,dtype=tf.float64), feat_value_reshape) # [None, field_size, 1]  Wi * Xi
y_first_order = tf.reduce_sum(input_tensor=w_mul_x, axis=2) # [None, field_size]

torch_y_first_order = torch.nn.Embedding(config.feature_size,1)(torch.from_numpy(feat_index).type(torch.LongTensor))
torch_w_mul_x = torch.mul(torch_y_first_order,torch_feat_value_reshape)
torch_y_first_order = torch.sum(torch_w_mul_x,dim=2)

# --------- 二维组合特征 ----------
embeddings = tf.multiply(tf.cast(embeddings_origin,dtype=tf.float64), feat_value_reshape) # [None, field_size, embedding_size] multiply不是矩阵相乘，而是矩阵对应位置相乘。这里应用了broadcast机制。
torch_embeddings = torch.mul(torch_embeddings_origin,torch_feat_value_reshape)

# sum_square part 先sum，再square
summed_features_emb = tf.reduce_sum(input_tensor=embeddings, axis=1) # [None, embedding_size]
summed_features_emb_square = tf.square(summed_features_emb)

torch_summed_features_emb = torch.sum(torch_embeddings,dim=1)
torch_summed_features_emb = torch.pow(torch_summed_features_emb,2)

# square_sum part
squared_features_emb = tf.square(embeddings)
squared_features_emb_summed = tf.reduce_sum(input_tensor=squared_features_emb, axis=1) # [None, embedding_size]

torch_squared_features_emb = torch.pow(torch_embeddings,2)
torch_squared_features_emb_summed = torch.sum(torch_squared_features_emb,dim=1)

# second order
y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_features_emb_summed)

torch_y_second_order = 0.5*torch.sub(torch_summed_features_emb,torch_squared_features_emb_summed)



# ----------- Deep Component ------------
y_deep = tf.reshape(embeddings_origin, shape=[-1, config.field_size * config.embedding_size]) # [None, field_size * embedding_size]
for i in range(0, len(deep_layers)):
    y_deep = tf.add(tf.matmul(y_deep, weights['layer_%d' % i]), weights['bias_%d' % i])
    y_deep = config.deep_layers_activation(y_deep)

torch_y_deep = torch.reshape(torch_embeddings_origin,[-1, config.field_size * config.embedding_size])
for i in range(0,len(deep_layers)):
    if i==0:
        torch_y_deep = torch.nn.Linear( config.field_size * config.embedding_size,deep_layers[0])(torch_y_deep)
        torch_y_deep = torch.nn.ReLU(inplace=True)(torch_y_deep)
    else:
        torch_y_deep = torch.nn.Linear(deep_layers[i-1],deep_layers[i])(torch_y_deep)
        torch_y_deep = torch.nn.ReLU(inplace=True)(torch_y_deep)


# ----------- output -----------
concat_input = tf.concat([y_first_order,y_second_order, tf.cast(y_deep,tf.float64)], axis=1)
out = tf.add(tf.matmul(concat_input, weights['concat_projection']), weights['concat_bias'])
out = tf.nn.sigmoid(out)

#########----------------torch  out
torch_concat_input = torch.cat([torch_y_first_order,torch_y_second_order,torch_y_deep],dim=1)
torch_out = torch.nn.Linear(torch_concat_input.shape[1],1)
torch_out = torch.nn.ReLU(inplace=True)(torch_out)


from torch import nn
class FM(nn.Module):
    def __init__(self,args):
        super(FM,self).__init__()
        # self.feat_index = args.feat_index
        self.feature_size = args['feature_size']
        self.embedding_size = args['embedding_size']
        self.field_size = args['field_size']
        self.em1 = nn.Embedding(self.feature_size,self.embedding_size)
        self.em2 = nn.Embedding(self.feature_size,1)

    def forward(self, feat_index,feat_value):
        torch_embeddings_origin = self.em1(feat_index)
        torch_feat_value_reshape = torch.reshape(feat_value, [-1, self.field_size, 1])
        ######一维特征
        torch_y_first_order = self.em2((feat_index))
        torch_w_mul_x = torch.mul(torch_y_first_order, torch_feat_value_reshape)
        torch_y_first_order = torch.sum(torch_w_mul_x, dim=2)
        ######二维特征
        torch_embeddings = torch.mul(torch_embeddings_origin, torch_feat_value_reshape)
        # sum_square part 先sum，再square
        torch_summed_features_emb = torch.sum(torch_embeddings, dim=1)
        torch_summed_features_emb = torch.pow(torch_summed_features_emb, 2)
        # square_sum part
        torch_squared_features_emb = torch.pow(torch_embeddings, 2)
        torch_squared_features_emb_summed = torch.sum(torch_squared_features_emb, dim=1)

        # second order
        torch_y_second_order = 0.5 * torch.sub(torch_summed_features_emb, torch_squared_features_emb_summed)

        return torch_y_first_order,torch_y_second_order


class Torch_Deep_FM(nn.Module):
    def __init__(self,args):
        super(Torch_Deep_FM,self).__init__()
        self.feature_size = args['feature_size']
        self.field_size = args['field_size']
        self.deep_layers = args['deep_layers']
        self.embedding_size = args['embedding_size']
        # self.feat_index = args.feat_index
        # self.feat_value = args.feat_value
        self.num_class = args['num_class']
        self.em1 = nn.Embedding(self.feature_size, self.embedding_size)
        self.em2 = nn.Embedding(self.feature_size, 1)

        self.fc0 = nn.Linear(self.field_size*self.embedding_size,self.deep_layers[0])
        self.relu = nn.ReLU(inplace=True)
        self.fm = FM(args)

    def forward(self,feat_index,feat_value):
        torch_embeddings_origin = self.em1(feat_index)
        torch_y_deep = torch.reshape(torch_embeddings_origin, [-1, self.field_size * self.embedding_size])
        for i in range(0, len(self.deep_layers)):
            if i == 0:
                torch_y_deep = self.fc0(torch_y_deep)
                torch_y_deep = self.relu(torch_y_deep)
            else:
                torch_y_deep = nn.Linear(self.deep_layers[i - 1], self.deep_layers[i])(torch_y_deep)
                torch_y_deep = self.relu(torch_y_deep)

        torch_y_first_order, torch_y_second_order = self.fm(feat_index,feat_value)
        torch_concat_input = torch.cat([torch_y_first_order, torch_y_second_order, torch_y_deep], dim=1)
        torch_out = nn.Linear(torch_concat_input.shape[1], self.num_class)
        # torch_out = self.relu(torch_out)

        return torch_out

################参数设置
args = {}
args['feature_size'] = fd.feat_dim
args['field_size'] = len(Xi_train[0])
args['embedding_size'] = 8
args['num_class'] = 2
args['deep_layers'] = [32,32]
args['lr'] = 0.001


#########----------------torch  out


config.loss = "logloss"
config.l2_reg = 0.1
config.learning_rate = 0.1

# loss
if config.loss == "logloss":
    loss = tf.losses.log_loss(label, out)

elif config.loss == "mse":
    loss = tf.losses.mean_squared_error(label, out)

# l2
if config.l2_reg > 0:
    loss += tf.contrib.layers.l2_regularizer(config.l2_reg)(weights['concat_projection'])
    for i in range(len(deep_layers)):
        loss += tf.contrib.layers.l2_regularizer(config.l2_reg)(weights['layer_%d' % i])

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Torch_Deep_FM(args).to(device)

from torch import optim
torch_optim = optim.Adam(model.parameters(),lr=args['lr'],weight_decay=0.95)
criterion = nn.CrossEntropyLoss()

def val(model,dataloader):
    """
    计算模型在验证集上的信息
    """
    model.eval()########固定
    acc_val = 0
    total = 0
    loss_val = 0
    val_iteration = 0
    for i,(images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        acc = (predicted == labels).sum().item() / len(labels)
        acc_val +=acc
        loss_val +=loss.item()
        val_iteration += 1

    acc_val /= val_iteration
    loss_val /= val_iteration

    model.train()####重启
    # print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    # print('loss of test images:%.3f' % (np.array(loss_my).mean()))
    return loss_val,acc_val


#
# Xi_train_, Xv_train_, y_train_
# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(
#     dataset=train_data,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=4
# )
# val_dataloader = DataLoader(
#     dataset=val_data,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=4
# )

loss_train = []
loss_val = []

args['epoch'] = 3
args['batch_size'] = 1024





for epoch in range(args['epoch']):
    shuffle_in_unison_scary(Xi_train_, Xv_train_, y_train_)
    total_batch = int((len(y_train_) - 1) / args['batch_size']) + 1
    for i in range(total_batch):
        Xi_batch, Xv_batch, y_batch = get_batch(Xi_train_, Xv_train_, y_train_, args['batch_size'], i)
        Xi_batch = torch.from_numpy(np.array(Xi_batch)).type(torch.LongTensor).to(device)
        Xv_batch = torch.from_numpy(np.array(Xv_batch)).type(torch.FloatTensor).to(device)
        y_batch = torch.from_numpy(np.array(y_batch).reshape(-1,1)).squeeze(1).to(device)

        outputs = model(Xi_batch,Xv_batch)
        loss = criterion(outputs,y_batch)
        _,predicted = torch.max(outputs.data,1)
        loss_train += loss.item()

        ###########backward and optimize
        torch_optim.zero_grad()
        loss.backward()
        torch_optim.step()

        if (i + 1) % 10 == 0:
            print('steps:[%d],train_loss:[%.3f]'%(i+1,loss.item()))
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},Acc: {:.3f}'
            #       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),acc))





#####训练

dfm_params = {
    "use_fm":True,
    "use_deep":True,
    "embedding_size":8,
    "dropout_fm":[1.0,1.0],
    "deep_layers":[32,32],
    "dropout_deep":[0.5,0.5,0.5],
    "deep_layer_activation":tf.nn.relu,
    "epoch":50,
    "batch_size":1024,
    "learning_rate":0.001,
    "optimizer":"adam",
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "l2_reg":0.01,
    "verbose":True,
    "eval_metric":gini_norm,
    "random_seed":config.RANDOM_SEED
}

###ddeep FM model
dfm_params['feature_size'] = fd.feat_dim
dfm_params['field_size'] = len(Xi_train[0])
dfm = my_DeepFM(**dfm_params)

# # ------------------ FM Model ------------------
# fm_params = dfm_params.copy()
# fm_params["use_deep"] = False
# dfm_fm = my_DeepFM(**fm_params)
#
# # ------------------ DNN Model ------------------
dnn_params = dfm_params.copy()
dnn_params["use_fm"] = False
dfm = my_DeepFM(**dnn_params)

# ------------------------------init session

sess = tf.Session(graph=tf.get_default_graph())

sess.run(tf.global_variables_initializer())

max_checks_without_progress = 10
checks_without_progress = 0
best_gini = 0

from time import time
from sklearn.metrics import roc_auc_score
has_valid = Xv_valid_ is not None
loss_train = 0
loss_test = 0
gini_train = []
gini_valid = []
for epoch in range(dfm.epoch):
    t1 = time()

    pre_train=[]
    pre_valid=[]

    shuffle_in_unison_scary(Xi_train_, Xv_train_, y_train_)
    total_batch = int((len(y_train_)-1) / dfm.batch_size)+1
    for i in range(total_batch):
        Xi_batch, Xv_batch, y_batch = get_batch(Xi_train_, Xv_train_, y_train_, dfm.batch_size, i)
        feed_dict = {dfm.feat_index: np.array(Xi_batch),
                       dfm.feat_value: np.array(Xv_batch),
                       dfm.label: np.array(y_batch).reshape((-1,1)),
                       dfm.dropout_keep_fm:dfm.dropout_fm,
                       dfm.dropout_keep_deep: dfm.dropout_deep,
                       dfm._training:True
                       }
        loss, opt, train_out= sess.run((dfm.loss, dfm.train_step,dfm.out), feed_dict=feed_dict)

        loss_train +=loss

    ##########----------------gini train---------
    for i in range(total_batch):
        dummy = [1] * len(Xi_train_)
        Xi_batch, Xv_batch, y_batch = get_batch(Xi_train_, Xv_train_, dummy, dfm.batch_size, i)
        num_batch = len(y_batch)
        feed_dict = {dfm.feat_index: np.array(Xi_batch),
                     dfm.feat_value: np.array(Xv_batch),
                     dfm.label: np.array(y_batch).reshape((-1, 1)),
                     dfm.dropout_keep_fm: [1.0] * len(dfm.dropout_fm),
                     dfm.dropout_keep_deep: [1.0] * len(dfm.dropout_deep),
                     dfm._training: False
                     }
        loss, train_out = sess.run((dfm.loss, dfm.out), feed_dict=feed_dict)
        if i == 0:
            pre_train = np.reshape(train_out, (num_batch,))
        else:
            pre_train = np.concatenate((pre_train, np.reshape(train_out, (num_batch,))))

    sig_gini_train = gini_norm(y_train_, pre_train)

    gini_train.append(sig_gini_train)


    feed_dict = {dfm.feat_index: np.array(Xi_valid_),
                 dfm.feat_value: np.array(Xv_valid_),
                 dfm.label: np.array(y_valid_).reshape((-1,1)),
                 dfm.dropout_keep_fm: [1.0] * len(dfm.dropout_fm),
                 dfm.dropout_keep_deep: [1.0] * len(dfm.dropout_deep),
                 dfm._training: False
                 }
    loss_test,  valid_out = sess.run((dfm.loss, dfm.out), feed_dict=feed_dict)
    pre_valid.append(valid_out)
    pre_valid = [y for x in pre_valid for y in x]
    sig_gini_valid = gini_norm(y_valid_, pre_valid)
    gini_valid.append(sig_gini_valid)

    if sig_gini_valid > best_gini:
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        best_params = {gvar.op.name:value for gvar,value in zip(gvars,sess.run(gvars))}
        best_gini = sig_gini_valid
        checks_without_progress = 0
    else:
        checks_without_progress += 1

    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                  % (epoch + 1, sig_gini_train, sig_gini_valid, time() - t1))

    # if checks_without_progress > max_checks_without_progress:
    #     print('early stopping!')
    #     break
##########将训练过程中保存的最好的参数重新返回到模型参数，此时得到的是最好的模型
if best_params:
    gvars_names = list(best_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in
                  gvars_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: best_params[gvar_name] for gvar_name in gvars_names}
    sess.run(assign_ops, feed_dict=feed_dict)




plt.figure()
plt.plot(gini_train,'-o',label='train')
plt.plot(gini_valid,'-*',label='val')
plt.xlabel('epoch')
plt.legend(loc='upper left')#图例位置



##########valid  预测
feed_dict = {dfm.feat_index: np.array(Xi_valid_),
                 dfm.feat_value: np.array(Xv_valid_),
                 dfm.label: np.array(y_valid_).reshape((-1,1)),
                 dfm.dropout_keep_fm: [1.0] * len(dfm.dropout_fm),
                 dfm.dropout_keep_deep: [1.0] * len(dfm.dropout_deep)
                 # dfm._training:False
                 }
_,  valid_out = sess.run((dfm.loss,dfm.out), feed_dict=feed_dict)

sig_gini_valid = gini_norm(y_valid_, valid_out)
print('valid data gini is %.4f:'%sig_gini_valid )

############----------关闭会话----------
sess.close()








