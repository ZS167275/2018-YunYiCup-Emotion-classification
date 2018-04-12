#coding:utf-8
import numpy as np
np.random.seed(42)
import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix, hstack
from wordbatch.models import FTRL, FM_FTRL


def get_org_data():
    # train_first = pd.read_csv('../data/train_first_trans.csv')
    # train_second = pd.read_csv('../data/train_second_trans.csv')
    # train_jieba_cut = pd.read_csv('../tmp/train_and_test_cut.csv', encoding='gbk')[['Id', 'cut_word']]
    # train_trans = pd.concat([train_first, train_second])
    #
    # print('data add 1 and 2')
    # train_1 = train_trans[train_trans['Score'] == 1]
    # train_2 = train_trans[train_trans['Score'] == 2]
    # train_trans = pd.concat([train_1, train_trans, train_1, train_2, train_1, train_2, train_1], axis=0)
    test_trans = pd.read_csv('../data/test_second_trans.csv')
    #
    # if not os.path.exists('tmp.csv'):
    #     train_and_test_trans = pd.concat([train_trans, test_trans])
    #     train_and_test_trans = pd.merge(train_and_test_trans, train_jieba_cut, on=['Id'])
    #     train_and_test_trans = train_and_test_trans.sample(frac=1)
    #     train_and_test_trans = train_and_test_trans.reset_index(drop=True)
    #     train_and_test_trans['cut_tran'] = train_and_test_trans['cut_word'] + train_and_test_trans['Trans']
    #     train_and_test_trans.to_csv('tmp.csv', index=False)

    # else:
    train_and_test_trans = pd.read_csv('tmp.csv')
    # del train_second, train_jieba_cut, train_trans
    train = train_and_test_trans[~train_and_test_trans['Id'].isin(test_trans['Id'].unique())]
    test = train_and_test_trans[train_and_test_trans['Id'].isin(test_trans['Id'].unique())]
    # Index(['Discuss', 'Id', 'Score', 'Trans', 'cut_word'], dtype='object')
    return train, test

def get_data():
    train,test = get_org_data()
    data = pd.concat([train, test])
    print('train %s test %s'%(train.shape,test.shape))
    print('train columns',train.columns)

    # 根据具体分布决定需要合并的类别
    y = []
    for i in list(train['Score']):
        y.append(i + 0.0 )

    print(np.array(y).reshape(-1,1)[:,0])
    return data,train.shape[0],np.array(y).reshape(-1,1)[:,0],test['Id']

# 预处理
def pre_process(name):
    data,nrw_train,y,test_id = get_data()
    print('TfidfVectorizer')
    tf = TfidfVectorizer(ngram_range=(1,8),analyzer='char')
    discuss_tf = tf.fit_transform(data[name])
    # hash feat 使用word级别的数据
    # ha = HashingVectorizer(ngram_range=(1,2),lowercase=False)
    # discuss_ha = ha.fit_transform(data[name])
    # data = hstack((discuss_tf,discuss_ha)).tocsr()
    data = discuss_tf.tocsr()
    return data[:nrw_train],data[nrw_train:],y,test_id

def xx_mse_s(y_true,y_pre):
    y_true = pd.DataFrame({'res':list(y_true)})
    y_true['res'] = y_true['res'].astype(int)
    y_pre = pd.DataFrame({'res':list(y_pre)})
    y_pre['res'] = y_pre['res'].astype(int)
    return 1 / ( 1 + mean_squared_error(y_true['res'],y_pre['res'].values)**0.5)

N = 4
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

for name in ['Trans', 'cut_word']:
    print(name)

    X,test,y,test_id = pre_process(name)

    kf = StratifiedKFold(n_splits=N, random_state=42)

    cv_pred = []
    kf = kf.split(X,y)
    xx_mse = []

    rg_1 = Ridge(solver='auto', fit_intercept=True , max_iter= 400, normalize=False, tol=0.001,random_state=42)
    rg_2 = Ridge(solver='auto', fit_intercept=True, max_iter=400, normalize=False, tol=0.0001, random_state=128)
    averaged_models = AveragingModels(models = (rg_1,rg_2))

    # fm_1 = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
    #                D_fm=200, e_noise=0.0001, iters=150, inv_link="identity")


    for i ,(train_fold,test_fold) in enumerate(kf):
        X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold]
        averaged_models.fit(X_train, label_train)
        val_ = averaged_models.predict(X_validate)
        print(xx_mse_s(label_validate, val_))
        cv_pred.append(averaged_models.predict(test))
        xx_mse.append(xx_mse_s(label_validate, val_))

    print('xx_result',np.mean(xx_mse))

    s = 0
    # for i in cv_pred:
    s = 0.2 * cv_pred[0] + 0.25*cv_pred[1] + 0.3*cv_pred[2] + 0.25*cv_pred[3]
    # s = s/N
    res = pd.DataFrame()
    res['Id'] = list(test_id)
    res['pre'] = list(s)

    res.to_csv('../result/%s_tfidf_20180408.csv'%(name),index=False,header=False)