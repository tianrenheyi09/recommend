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








