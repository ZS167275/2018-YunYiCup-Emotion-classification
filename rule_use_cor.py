#coding:utf-8
import jieba
import pandas as pd
import os
import re
# 正则表达式
zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
def contain_zh(word):
    '''
    判断传入字符串是否包含中文
    :param word: 待判断字符串
    :return: True:包含中文  False:不包含中文
    '''
    # word = word.decode()
    global zh_pattern
    match = zh_pattern.search(word)
    if match is None:
        return 1
    return 0


def remove_chinese(word):
    '''
    过滤特殊符号
    :param word:
    :return:
    '''
    word = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", word)

    # str = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", word)
    return word

train_first = pd.read_csv('../data/train_first.csv')
train_second = pd.read_csv('../data/train_second.csv')
train = pd.concat([train_first,train_second]).reset_index(drop=True)
test = pd.read_csv('../data/predict_second.csv')
train_and_test = pd.concat([train,test])

if not os.path.exists('../tmp/train_and_test_cut.csv'):
    train_and_test['cut_word'] = train_and_test['Discuss'].apply(
        lambda x:" ".join(jieba.cut(str(x)))
    )
    train_and_test.to_csv('../tmp/train_and_test_cut.csv',index=False)
else:
    train_and_test = pd.read_csv('../tmp/train_and_test_cut.csv',encoding='gbk')

from sklearn.preprocessing import LabelEncoder

train = train_and_test[train_and_test['Id'].isin(train['Id'].unique())]
test = train_and_test[~train_and_test['Id'].isin(train['Id'].unique())]


train['is_have_chinese'] = train['Discuss'].apply(contain_zh)
test['is_have_chinese'] = test['Discuss'].apply(contain_zh)

train['remove_chinese'] = train['Discuss'].apply(remove_chinese)
test['remove_chinese'] = test['Discuss'].apply(remove_chinese)

# not_chinese_train = train[train['is_have_chinese']==1]
# not_chinese_train = not_chinese_train.groupby(['Discuss','Score']).Id.size().reset_index()
# print(not_chinese_train)
#
# not_chinese_test = test[test['is_have_chinese']==1]
# not_chinese_test = not_chinese_test.groupby(['Discuss']).Id.size().reset_index()
# print(not_chinese_test)
#
# print(train[train['is_have_chinese']==1])
# print(test[test['is_have_chinese']==1])


# 拆分train
# train['remove_chinese'] = train['Discuss'].apply(lambda x:str(x).split("，")[0])

# sub_index = []
# sub_score = []
# for sub_discuss in test.iterrows():
#     sub_discuss = sub_discuss[1]
#     tmp_discuss = sub_discuss['Discuss']
#     if len(tmp_discuss) >= 1:
#         tmp_train = train[train['Discuss'].str.startswith(tmp_discuss,na=False)]
#         score = tmp_train.groupby(['Score']).Id.size().reset_index().sort_values('Id',ascending=False)
#         score.rename(columns={'Id': 'prob_score', 'Score': 'Score_'}, inplace=True)
#         score['r'] = 1
#         score = pd.DataFrame(score).drop_duplicates(['r'])
#         score = score['Score_'].values
#         if len(score) == 0:
#             pass
#         else:
#             sub_index.append(sub_discuss['Id'])
#             sub_score.append(score[0])
#
# tmp_score = pd.DataFrame()
# tmp_score['Id'] = list(sub_index)
# tmp_score['Score'] = list(sub_score)
# print(tmp_score.shape)
# tmp_score.to_csv('../tmp/tmp_score_start_with.csv',index=False)

if True:
    t_tmp = train.groupby(['remove_chinese','Score']).Id.size().reset_index().sort_values('Id',ascending=False)
    t_tmp.rename(columns={'Id':'prob_score','Score':'Score_'},inplace=True)
    t_tmp = pd.DataFrame(t_tmp).drop_duplicates(['remove_chinese'])
    test_in_train = pd.merge(test,t_tmp,on=['remove_chinese'],how='inner')

    # print(test_in_train)

# print(test_in_train[test_in_train['is_have_chinese']==1])

# exit()
best_score = pd.read_csv(u'../result/stacking_2.csv',header=None)
best_score.columns = ['Id','Score']

ruler_2 = pd.read_csv('../tmp/tmp_score_start_with.csv')
ruler_2.rename(columns={'Score':'Score_2'},inplace=True)

best_score = pd.merge(best_score,test_in_train[['Id','Score_','Discuss']],how='left',on=['Id'])
best_score = pd.merge(best_score,ruler_2,how='left',on=['Id'])

# best_score['Score_'] = best_score['Score_'].fillna(best_score['Score_2'])


# best_score['Score_'] = best_score['Score_'].fillna(best_score['Score'])

best_score['Score_2'] = best_score['Score_2'].fillna(best_score['Score'])
best_score['Score_'] = best_score['Score_'].fillna(best_score['Score'])

best_score['Score_'] = (best_score['Score_'] *0.8+ best_score['Score_2']*0.2)



# best_score = best_score[['Id','Score','Score_','Score_2']]
# print(best_score)

best_score[['Id','Score_']].to_csv('../result/stacking_1_04_08_2.csv',index=False,header=False)
