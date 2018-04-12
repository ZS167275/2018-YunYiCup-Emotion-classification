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
        # if i == 5:
        #     i = 6
        # else:
        #     i = i
        # y.append(i + np.random.random(1)* 2/4 )
        y.append(i + 0.0 )

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
    # tr4w = TextRank4Keyword()
    # start = time.time()
    # print('split_word',)
    # data['cut_comment'] = data['Discuss'].apply(
    #     lambda x:' '.join(i for i in split_(tr4w,x)))
    # print('split_word_end',(time.time() - start)/60000 )

    # with open('../data/stop_word.txt', encoding='GBK') as words:
    #     stop_words = [i.strip() for i in words.readlines()]
    # # with open('../data/stop_word_2.txt', encoding='UTF-8') as words:
    #     # stop_words_2 = [i.strip() for i in words.readlines()]
    # # stop_words = list(set(stop_words + stop_words_2))
    # # 过滤停用词
    # data['cut_comment'] = data['Discuss'].apply(lambda x:' '.join(i for i in jieba.cut(x) if i not in stop_words))
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

    # length = csr_matrix(pd.get_dummies(data[['length','unique_count_log']],sparse=True).values)
    # add_feat = csr_matrix(data[['is_again_1','is_again_2','is_again_3','is_first_1','is_first_2']].values)
    # 归一化
    # mm_1 = MinMaxScaler()
    # length = mm_1.fit_transform(data['length'].values.reshape(-1,1))
    # mm_2 = MinMaxScaler()
    # unique_count = mm_2.fit_transform(data['unique_count'].values.reshape(-1,1))

    # data = hstack((length,unique_count,discuss_tf,discuss_ha,add_feat)).tocsr()
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

# import lightgbm as lgb
#
# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __init__(self, models):
#         self.models = models
#
#     # we define clones of the original models to fit the data in
#     def fit(self, X, y):
#         self.models_ = [clone(x) for x in self.models]
#
#         # Train cloned base models
#         for model in self.models_:
#             model.fit(X, y)
#
#         return self
#
#     # Now we do the predictions for cloned models and average them
#     def predict(self, X):
#         predictions = np.column_stack([
#             model.predict(X) for model in self.models_
#         ])
#         return np.mean(predictions, axis=1)
# import keras as ks
import tensorflow as tf
# from keras.models import Model
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF

# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu': 0})))

# mlp
# def fit_predict(X_train,y_train,X_test):
#     # config = tf.ConfigProto(
#     #     intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
#     with tf.Session(graph=tf.Graph()) as sess:
#         ks.backend.set_session(sess)
#         model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
#         out = ks.layers.Dense(192, activation='relu')(model_in)
#         out = ks.layers.Dense(64, activation='relu')(out)
#         out = ks.layers.Dense(64, activation='relu')(out)
#         out = ks.layers.Dense(1)(out)
#         # model = ks.models()
#         model = ks.models.Model(model_in, out)
#         model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
#         for i in range(3):
#             model.fit(x=X_train, y=y_train, batch_size=2**(5 + i), epochs=1, verbose=0)
#         return model.predict(X_test)[:, 0]
import lightgbm as lgb
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline

# lgb_params = {
#               'max_depth': 11,
#               'learning_rate': 0.05,
#               'objective': 'regression',
#               'metric': 'rmse',
#               'bagging_fraction': 0.90, # Best one so far - [884]       valid_0's rmse: 0.434035 - 0.89
#               'colsample_bytree': 0.90,
#               'num_leaves':  2 ** 11,  # 8192
#               'num_threads': 8,
#               'min_child_weight': 5,
#               'bagging_seed': 0,
#               # 'feature_fraction_seed': 0,
#              }

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

# from sklearn.naive_bayes import GaussianNB
# averaged_models = AveragingModels(models = (model_1,lg))
# model_1 = GaussianNB()

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


# 分层
# 4.7 归一到 5
# 0.526861236399
# 0.52682773802
# 0.526292337313
# 0.526072111603
# xx_result 0.526513355834 5225 0.52182

# 增加 snom的预测结果
# 0.526947803501
# 0.526617776608
# 0.526526435843
# 0.526139719229
# xx_result 0.526557933795 0.52292  0.51188


# test rank
# 0.528647657041
# 0.528802056321
# 0.527715836747
# 0.527729524105
# xx_result 0.528223768553 5267

# 0.52621

# text_cnn + trad

# 0.528854669024
# 0.528638975359
# 0.527977496462
# 0.527139551779
# xx_result 0.528152673156 0.52617

# jieba no anything
# 0.529509264083
# 0.529191780346
# 0.528089800046
# 0.528159739354
# xx_result 0.528737645957 0.52595

# 0.526706797666
# 0.52558534458
# 0.525715084183
# 0.525900180434
# xx_result 0.525976851716


# 使用char的特征计算
# 0.527220253028
# 0.526278860359
# 0.525519165066
# 0.524849070442
# xx_result 0.525966837224 51876




# xx_result 0.529966884026


# stacking rgide + lasso = lr
# 修改4-5 7 / 修改 1 2 3 4 5 7
# xx_result 0.526154094516 0.51832 0.522