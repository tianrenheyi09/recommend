import pandas as pd



movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
data1 = pd.merge(movies,ratings,on='movieId')
print('len of movies:',len(movies))
print('len if ratings:',len(ratings))
print('len of data:',len(data1))

data1[['userId','rating','movieId','title']].sort_values(by='userId').to_csv('data/data.csv',index=False)
########利用字典表示用户电影评分
file = open('data/data.csv')
data = {}
for line in file.readlines()[1:-1]:
    line = line.strip().split(',')
    if not line[0] in data.keys():
        data[line[0]] ={line[3]:line[1]}
    else:
        data[line[0]][line[3]] = line[1]

print(data)

#########计算欧氏距离以及两者之间的相似度
from math import *
def Euclidean(user1,user2):
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    for key in user1_data.keys():
        if key in user2_data.keys():
            distance += pow(float(user1_data[key])-float(user2_data[key]),2)

    return 1/(1+sqrt(distance))

def top10_simliar(userID):
    res = []
    for userid in data.keys():
        if not userid == userID:
            simliar = Euclidean(userID,userid)
            res.append((userid,simliar))
    res.sort(key=lambda val:val[1])
    return res[:4]

RES = top10_simliar('1')
print(RES)

def recommend(user):
    top_sim_user = top10_simliar(user)[0][0]
    items = data[top_sim_user]
    recommenddations = []
    for item in items.keys():
        if item not in data[user].keys():
            recommenddations.append((item,items[item]))
    recommenddations.sort(key = lambda val:val[1],reverse=True)
    return recommenddations[:10]

Recommend = recommend('1')
print(Recommend)

######计算pearson相关系数
def pearson_sim(user1,user2):
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    common = {}
    ####找到两位用户都评论过的电影
    for key in user1_data.keys():
        if key in user2_data.keys():
            common[key] = 1
    if len(common)==0:
        return 0
    n = len(common)
    print(n,common)
    ###计算得分和
    sum1 = sum([float(user1_data[movie]) for movie in common])
    sum2 = sum([float(user2_data[movie]) for movie in common])
    ####计算评分平方和
    sum1sq = sum([pow(float(user1_data[movie]),2) for movie in common])
    sum2sq = sum([pow(float(user2_data[movie]),2) for movie in common])
    #####计算乘积和
    psum= sum([float(user1_data[it])*float(user2_data[it]) for it in common])
    #####相关系数
    num = psum-(sum1*sum2)/n
    den = sqrt((sum1sq-pow(sum1,2)/n)*(sum2sq-pow(sum2,2)/n))
    if den == 0:
        return 0
    r = num/den
    return r

R = pearson_sim('1','3')
print(R)
