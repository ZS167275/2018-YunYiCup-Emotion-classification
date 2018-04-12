#coding:utf-8
import numpy as np
np.random.seed(42)
import snownlp
import pandas as pd
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import scipy
from sklearn.model_selection import KFold,StratifiedKFold
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

import lightgbm as lgb
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline

def get_data():
    train = pd.read_csv('../data/train_first.csv')
    test = pd.read_csv('../data/predict_first.csv')
    data = pd.concat([train, test])
    print('train %s test %s'%(train.shape,test.shape))
    print('train columns',train.columns)

    # 根据具体分布决定需要合并的类别
    y = []
    for i in list(train['Score']):
        # 归类处理
        if i == 1 | i==2:
            i = 1
        else:
            i = i
        y.append(i + np.random.random(1)* 2/4 )
#         y.append(i + 0.0 )

    print(np.array(y).reshape(-1,1)[:,0])
    return data,train.shape[0],np.array(y).reshape(-1,1)[:,0],test['Id']

from textrank4zh import TextRank4Keyword, TextRank4Sentence
def split_(tr4w,x):
    # tr4w.analyze(text=x, lower=True, window=4)
    result = tr4w.seg.segment(x,lower=True)
    s = []
    for words in result.words_no_filter:
        s = s + words
    return s
import time
# 分词处理
def split_discuss(data,nrw_train):
    data['cut_comment'] = data['Discuss'].apply(lambda x:' '.join(i for i in jieba.cut(x)))
    # 分词后的长度信息

    return data


def snow_score(data):
    return snownlp.SnowNLP(data).sentiments

# 预处理
def pre_process():
    data,nrw_train,y,test_id = get_data()
    # print('snownlp')
    # snow_score_mat = np.array(data['Discuss'].apply(snow_score))
    # snow_score_scp = csr_matrix(snow_score_mat.reshape(-1,1))
    print('vectory')
    data = split_discuss(data,nrw_train)
    # cv = CountVectorizer(ngram_range=(1,2))
    # discuss = cv.fit_transform(data['Discuss'])
    print('TfidfVectorizer')
    tf = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
    discuss_tf = tf.fit_transform(data['cut_comment'])
    # hash feat 使用word级别的数据
    ha = HashingVectorizer(ngram_range=(1,2),lowercase=False)
    discuss_ha = ha.fit_transform(data['cut_comment'])
    
    data = hstack((discuss_tf,discuss_ha)).tocsr()

    return data[:nrw_train],data[nrw_train:],y,test_id

def xx_mse_s(y_true,y_pre):
    y_true = pd.DataFrame({'res':list(y_true)})
    y_true['res'] = y_true['res'].astype(int)

    y_pre = pd.DataFrame({'res':list(y_pre)})
    y_pre['res'] = y_pre['res'].astype(int)

    return 1 / ( 1 + mean_squared_error(y_true['res'],y_pre['res'].values)**0.5)


N = 4
X,test,y,test_id = pre_process()
kf = StratifiedKFold(n_splits=N, random_state=42)
# kf = KFold(n_splits=N,shuffle=False,random_state=42)
cv_pred = []
kf = kf.split(X,y)
xx_mse = []


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


lr = LinearRegression()
rg_1 = Ridge(solver='auto', fit_intercept=True , max_iter= 400, normalize=False, tol=0.001,random_state=42)
rg_2 = Ridge(solver='auto', fit_intercept=True , max_iter= 300, normalize=False, tol=0.0001,random_state=128)
rg_3 = Ridge(solver='auto', fit_intercept=True , max_iter= 200, normalize=False, tol=0.001,random_state=256)
lasso_1 = Lasso(random_state=42, max_iter= 1000,tol=0.001)
lasso_2 = Lasso(random_state=128, max_iter= 500,tol=0.0001)
ENet = ElasticNet(random_state=42,max_iter= 1000,tol=0.001)

model_1 = StackingRegressor(regressors=[rg_1,rg_2, rg_3,lasso_1,lasso_2], meta_regressor=lr,verbose=3)

for i ,(train_fold,test_fold) in enumerate(kf):
    X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold]
    model_1.fit(X_train, label_train)
    # val_ = fit_predict(X_train,label_train,X_validate)
    val_ = model_1.predict(X_validate)
    print(xx_mse_s(label_validate, val_))

    cv_pred.append(model_1.predict(test))
    # cv_pred.append(fit_predict(X_train,label_train,test))
    xx_mse.append(xx_mse_s(label_validate, val_))

import numpy as np
print('xx_result',np.mean(xx_mse))

s = 0
for i in cv_pred:
    s = s + i

s = s/N
res = pd.DataFrame()
res['Id'] = list(test_id)
res['pre'] = list(s)
# res['pre'] = res['pre'].astype(int)

res.to_csv('../result/0.51832.csv',index=False,header=False)
