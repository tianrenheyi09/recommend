import os
import numpy as np
import pandas as pd
from DataReader import FeatureDictionary, DataParser
from matplotlib import pyplot as plt
import config
from metrics import gini_norm
import random
import time

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



import torch
from torch.utils.data import Dataset,DataLoader

class deal_data(Dataset):
    def __init__(self,Xi,Xv,y):
        self.xi_data = torch.from_numpy(np.array(Xi)).type(torch.LongTensor)
        self.xv_data = torch.from_numpy(np.array(Xv)).type(torch.FloatTensor)
        self.y_data = torch.from_numpy(np.array(y)).type(torch.FloatTensor)

        self.len = len(y)

    def __getitem__(self,index):
        return self.xi_data[index],self.xv_data[index],self.y_data[index]

    def __len__(self):
        return self.len




import torch
from torch import nn
class FM(nn.Module):
    def __init__(self,args):
        super(FM,self).__init__()
        # self.feat_index = args.feat_index
        self.feature_size = args['feature_size']
        self.embedding_size = args['embedding_size']
        self.field_size = args['field_size']
        self.em1 = nn.Embedding(self.feature_size,self.embedding_size)
        nn.init.normal_(self.em1.weight.data,mean=0.0,std=0.01)

        self.em2 = nn.Embedding(self.feature_size,1)
        nn.init.normal_(self.em2.weight.data,mean=0,std=0.1)


    def forward(self, feat_index,feat_value):
        torch_embeddings_origin = self.em1(feat_index)####shape:batch_size*filed_size*embedding_size
        torch_feat_value_reshape = torch.reshape(feat_value, [-1, self.field_size, 1])
        ######一维特征
        torch_y_first_order = self.em2((feat_index))####shape:batch_size*filed_size*1
        torch_w_mul_x = torch.mul(torch_y_first_order, torch_feat_value_reshape)
        torch_y_first_order = torch.sum(torch_w_mul_x, dim=2)####shape:batch_size*filed_size
        ######二维特征
        torch_embeddings = torch.mul(torch_embeddings_origin, torch_feat_value_reshape)####shape:batch_size*filed_size*embedding_size
        # sum_square part 先sum，再square
        torch_summed_features_emb = torch.sum(torch_embeddings, dim=1)####shape batch_size*embedding_size
        torch_summed_features_emb = torch.pow(torch_summed_features_emb, 2)####shape batch_size*embedding_size
        # square_sum part
        torch_squared_features_emb = torch.pow(torch_embeddings, 2)####shape batch_size*filed_size*embedding_size
        torch_squared_features_emb_summed = torch.sum(torch_squared_features_emb, dim=1)####shape batch_size*embedding_size


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

        self.dropout_keep = args['dropout_keep']
        self.is_batch_norm = args['is_batch_norm']
        self.is_deep_dropout = args['is_deep_dropout']

        self.use_fm = args['use_fm']
        self.use_deep = args['use_deep']

        self.num_class = args['num_class']
        self.em1 = nn.Embedding(self.feature_size, self.embedding_size)
        nn.init.normal_(self.em1.weight.data, mean=0.0, std=0.01)
        self.em2 = nn.Embedding(self.feature_size, 1)
        nn.init.normal_(self.em2.weight.data, mean=0, std=0.1)

        self.relu = nn.ReLU(inplace=True)
        self.fm = FM(args)

        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2 / (input_size + self.deep_layers[0]))

        self.fc0 = nn.Linear(self.field_size * self.embedding_size, self.deep_layers[0])
        if self.is_batch_norm:
            self.batch_norm_0 = nn.BatchNorm1d(self.deep_layers[0])
        if self.is_deep_dropout:
            self.fc0_0_dropout = nn.Dropout(self.dropout_keep[1])
        nn.init.normal_(self.fc0.weight,mean=0.0,std=glorot)
        nn.init.normal_(self.fc0.bias.data,mean=0.0,std=glorot)

        for i in range(1,len(self.deep_layers)):
            glorot = 2 / (self.deep_layers[i - 1] + self.deep_layers[i])
            setattr(self,'fc_'+str(i),nn.Linear(self.deep_layers[i-1],self.deep_layers[i]))
            nn.init.normal_(getattr(self,'fc_'+str(i)).weight.data,mean=0,std =glorot)
            nn.init.normal_(getattr(self,'fc_'+str(i)).bias.data,mean=0,std=glorot)

            if self.is_batch_norm:
                setattr(self,'batch_norm_'+str(i),nn.BatchNorm1d(self.deep_layers[i]))
            if self.is_deep_dropout:
                setattr(self,'fc_'+str(i)+'_dropout',nn.Dropout(self.dropout_keep[i+1]))

        print('init deep layers succeed')
        #########concat
        if self.use_fm and self.use_deep:
            input_size = self.field_size+self.embedding_size+self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size+self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0 / (input_size + 1))

        self.fc_last = nn.Linear(input_size,1)
        nn.init.normal_(self.fc_last.weight.data,mean=0,std=glorot)
        nn.init.constant_(self.fc_last.bias.data,0.01)

    def forward(self,feat_index,feat_value):
        torch_embeddings_origin = self.em1(feat_index)
        torch_value_origin = torch.reshape(feat_value, [-1,self.field_size, 1])
        torch_embeddings_origin = torch.mul(torch_embeddings_origin, torch_value_origin)#####共享embedding
        torch_y_deep = torch.reshape(torch_embeddings_origin, [-1, self.field_size * self.embedding_size])
        for i in range(0, len(self.deep_layers)):
            if i == 0:
                torch_y_deep = self.fc0(torch_y_deep)
                if self.is_batch_norm:
                    torch_y_deep = self.batch_norm_0(torch_y_deep)
                torch_y_deep = self.relu(torch_y_deep)
                if self.is_deep_dropout:
                    torch_y_deep = self.fc0_0_dropout(torch_y_deep)
            else:
                torch_y_deep = getattr(self,'fc_'+str(i))(torch_y_deep)
                if self.is_batch_norm:
                    torch_y_deep = getattr(self,'batch_norm_'+str(i))(torch_y_deep)
                torch_y_deep = self.relu(torch_y_deep)
                if self.is_deep_dropout:
                    torch_y_deep = getattr(self,'fc_'+str(i)+'_dropout')(torch_y_deep)


        torch_y_first_order, torch_y_second_order = self.fm(feat_index,feat_value)
        if self.use_deep and self.use_fm:
            torch_concat_input = torch.cat([torch_y_first_order, torch_y_second_order, torch_y_deep], dim=1)
        elif self.use_deep:
            torch_concat_input = torch_y_deep
        elif self.use_fm:
            torch_concat_input = torch.cat([torch_y_first_order, torch_y_second_order], dim=1)

        torch_out = self.fc_last(torch_concat_input)

        return torch_out.squeeze(1)

def seed_torch(seed=2018):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def val(model,dataloader):
    """
    计算模型在验证集上的信息
    """
    model.eval()########固定
    total = 0
    loss_val = 0
    val_iteration = 0
    y_true = []
    y_pre = []
    for i, (Xi_batch, Xv_batch, y_batch) in enumerate(dataloader):
        Xi_batch = Xi_batch.to(device)
        Xv_batch = Xv_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(Xi_batch,Xv_batch)
        loss = criterion(outputs, y_batch)
        # _, predicted = torch.max(outputs.data, 1)
        y_true.append(y_batch.data.cpu().numpy())
        prob = F.sigmoid(outputs)
        y_pre.append(prob.data.cpu().numpy())

        loss_val +=loss.item()
        val_iteration += 1

    loss_val /= val_iteration

    y_true = np.concatenate(y_true, axis=0)
    y_pre = np.concatenate(y_pre, axis=0)
    gini_val = gini_norm(y_true, y_pre)

    model.train()####重启
    return loss_val,gini_val


def pre_pre(model,data_loader):
    y_pre = []
    model.eval()
    for i,(Xi_batch, Xv_batch, y_batch) in enumerate(data_loader):
        Xi_batch = Xi_batch.to(device)
        Xv_batch = Xv_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(Xi_batch,Xv_batch)
        prob = F.sigmoid(outputs)
        y_pre.append(prob.data.cpu().numpy())
    y_pre = np.concatenate(y_pre, axis=0)
    y_predict = [1 if mm>0.5 else 0 for mm in y_pre]
    return y_predict

def pre_prob(model,data_loader):
    y_pre = []
    model.eval()
    for i, (Xi_batch, Xv_batch, y_batch) in enumerate(data_loader):
        Xi_batch = Xi_batch.to(device)
        Xv_batch = Xv_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(Xi_batch, Xv_batch)
        prob = F.sigmoid(outputs)
        y_pre.append(prob.data.cpu().numpy())
    y_pre = np.concatenate(y_pre, axis=0)

    return y_pre


##########################--------------------------mian函数---------------------------
# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()

##########得到index和value
fd = FeatureDictionary(dfTrain=dfTrain,dfTest=dfTest, numeric_cols=config.NUMERIC_COLS,ignore_cols=config.IGNORE_COLS)
data_parser = DataParser(feat_dict=fd)
# Xi_train ：列的序号
# Xv_train ：列的对应的值
Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)
y_test = [1 for i in range(len(Xv_test))]
# print(dfTrain.dtypes)

# ############随机打乱划分训练集和验证集
np.random.seed(2018)
shuftle_index = np.random.permutation(len(Xi_train))
train_idx = shuftle_index[0:int(0.08*len(Xi_train))]
valid_idx = shuftle_index[int(0.08*len(Xi_train)):int(0.1*len(Xi_train))]
test_idx = shuftle_index[0:int(0.02*len(Xi_test))]
_get = lambda x,l:[x[i] for i in l]
Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
Xi_test_,Xv_test_,y_test_ = _get(Xi_test,test_idx),_get(Xv_test,test_idx),_get(y_test,test_idx)

train_data = deal_data(Xi_train_,Xv_train_,y_train_)
valid_data = deal_data(Xi_valid_,Xv_valid_,y_valid_)

test_data = deal_data(Xi_test_,Xv_test_,y_test_)

train_dataloader = DataLoader(dataset=train_data,batch_size=1024,shuffle=True)
valid_dataloader = DataLoader(dataset=valid_data,batch_size=1024,shuffle=False)
test_dataloader = DataLoader(dataset=test_data,batch_size=1024,shuffle=False)

print('数据loader完毕')
#######################3-------------------------------################参数设置
args = {}
args['feature_size'] = fd.feat_dim
args['field_size'] = len(Xi_train[0])
args['embedding_size'] = 8
args['num_class'] = 1
args['deep_layers'] = [32,32]
args['lr'] = 0.001
args['epoch'] = 10
args['batch_size'] = 1024

args['dropout_keep'] = [0.5, 0.5, 0.5]
args['is_batch_norm'] = True
args['is_deep_dropout'] = True
args['use_fm'] = True
args['use_deep'] = True

print('参数设置完毕')

#########----------------torch  out


seed_torch(seed=2018)#################################此函数可以保证训练的loss可以复现

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Torch_Deep_FM(args).to(device)

from torch import optim
import torch.nn.functional as F
torch_optim = optim.Adam(model.parameters(),lr=args['lr'])
criterion = F.binary_cross_entropy_with_logits

print('model 初始化 完成')
############-------------------------------------------------------------------------
print('start  training')

check_file = 'best_model.pkl'
history = {}
history['loss_train'] = []
history['loss_val'] = []
history['gini_train'] = []
history['gini_val' ] =[]
max_checks_without_progress = 5
checks_without_progress = 0
best_loss = np.infty
model.train()

for epoch in range(10):
    loss_train = 0.0

    steps = 0
    y_true = []
    y_pre = []
    t1 = time.time()
    for i,(Xi_batch,Xv_batch,y_batch) in enumerate(train_dataloader):
        Xi_batch = Xi_batch.to(device)
        Xv_batch = Xv_batch.to(device)
        y_batch = y_batch.to(device)
        y_true.append(y_batch.data.cpu().numpy())
        steps +=1
        outputs = model(Xi_batch, Xv_batch)

        # prob,_ = torch.max(nn.functional.softmax(outputs,1),1)
        prob  = F.sigmoid(outputs)
        y_pre.append(prob.data.cpu().numpy())

        loss = criterion(outputs, y_batch)
        # _, predicted = torch.max(outputs.data, 1)
        loss_train += loss.item()
        ###########backward and optimize
        torch_optim.zero_grad()
        loss.backward()
        torch_optim.step()
        # if (i + 1) % 10 == 0:
        #     print('steps:[%d],train_loss:[%.3f]' % (i + 1, loss.item()))

    y_true = np.concatenate(y_true,axis=0)
    y_pre = np.concatenate(y_pre,axis=0)
    gini_train = gini_norm(y_true,y_pre)

    loss_train /= steps
    loss_val,gini_val = val(model,valid_dataloader)
    history['loss_train'].append(loss_train)
    history['loss_val'].append(loss_val)
    history['gini_train'].append(gini_train)
    history['gini_val'].append(gini_val)

    if loss_val < best_loss:
        torch.save(model.state_dict(), check_file)
        best_loss = loss_val
        checks_without_progress = 0
    else:
        checks_without_progress += 1
        if checks_without_progress > max_checks_without_progress:
            print('early stopping!')
            break

    t2 = time.time()
    print('epoch:%d,time:%.3f,loss_train:%.3f,loss_val:%.3f,gini_train:%.3f,gini_val:%.3f'%(epoch+1,t2-t1,loss_train,loss_val,gini_train,gini_val))

print('\ntraining  fininshed')

#####################-------evalution  show
print('result show here')
plt.figure()
plt.plot(history['loss_train'],'-o',label='train')
plt.plot(history['loss_val'],'-*',label='val')
plt.xlabel('epoch')
plt.legend(loc='upper left')#图例位置
plt.show()

plt.figure()
plt.plot(history['gini_train'],'-o',label='gini_train')
plt.plot(history['gini_val'],'-*',label='gini_val')
plt.xlabel('epoch')
plt.legend(loc='upper right')#图例位置
plt.show()

#
# #####################加载训练过程中的最优参数
# print('加载最优模型参数')
# the_model = Torch_Deep_FM(args).to(device)
# the_model.load_state_dict(torch.load(check_file))
# ###模型的参数个数
# num_param = 0
# load_state = torch.load(check_file)
# for name,value in load_state.items():
#     print(name,' ',value.shape)
#     num_param +=1
# print('total num params is:',num_param)
#
# ############test阶段
# print('start testing')
# pre = pre_pre(the_model,test_dataloader)
# pre_prob = pre_prob(the_model,test_dataloader)


