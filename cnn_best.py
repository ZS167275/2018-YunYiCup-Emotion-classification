# -*- coding: utf-8 -*-
import pandas  as pd
import numpy as np
from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
import jieba
import jieba.posseg
import jieba.analyse
import codecs
from keras.layers import Input, Concatenate
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

train = pd.read_csv('data/train_first.csv')
train_1 = train[train['Score']==1]
train_2 = train[train['Score']==2]
train = pd.concat([train_1,train,train_1,train_2,train_1,train_2,train_1],axis=0)
train = train.sample(frac=1)
train = train.reset_index(drop=True)
test = pd.read_csv('data/predict_first.csv')

max_features = 10000 ## 词汇量
maxlen = 150  ## 最大长度
embed_size = 256 # emb 长度

def score(y_true, y_pred):
    return 1.0/(1.0+K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1)))

def splitWord(query, stopwords):
    wordList = jieba.cut(query)
    num = 0
    result = ''
    for word in wordList:
        word = word.rstrip()
        word = word.rstrip('"')

        #if word not in stopwords:
        if num == 0:
            result = word
            num = 1
        else:
            result = result + ' ' + word

      
       #result = result + ' ' + word
    return str(result.encode('utf-8'))
def preprocess(data):
    stopwords = {}
    for line in codecs.open('data/stop_word.txt','r',encoding='gbk'):
        stopwords[line.rstrip()]=1    
    data['doc'] = data['Discuss'].map(lambda x:splitWord(x,stopwords))
    return data

train.Discuss.fillna('_na_',inplace=True)
test.Discuss.fillna('_na_',inplace=True)
train = preprocess(train)
test = preprocess(test)

comment_text = np.hstack([train.doc.values])

tok_raw = Tokenizer(num_words=max_features)

tok_raw.fit_on_texts(comment_text)


train['Discuss_seq'] = tok_raw.texts_to_sequences(train.doc.values)
test['Discuss_seq'] = tok_raw.texts_to_sequences(test.doc.values)

def get_keras_data(dataset): 
    X={
        'Discuss_seq':pad_sequences(dataset.Discuss_seq,maxlen=maxlen)
    }
    return X

def cnn():
    #Inputs
    comment_seq = Input(shape=[maxlen],name='Discuss_seq')
    
    #Embeddings layers
    emb_comment =Embedding(max_features, embed_size)(comment_seq)
    
    # conv layers
    convs = []
    filter_sizes = [64,32,16,8,4]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=256,kernel_size=fsz,activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen-fsz+1)(l_conv)

        #avg_pool = GlobalAveragePooling1D()(l_conv)
        #max_pool = GlobalMaxPooling1D()(l_conv)
        #conc = concatenate([avg_pool, max_pool])

        l_pool = Flatten()(l_pool)
        convs.append(l_pool)

    merge =concatenate(convs,axis=1)
    
    out = Dropout(0.25)(merge)
    output  = Dense(32,activation='relu')(out)
    
    output = Dense(units=1,activation='linear')(output)
    
    model = Model([comment_seq],output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", score])
    return model

X_train =get_keras_data(train)
X_train = X_train['Discuss_seq']


X_test = get_keras_data(test)
X_test = X_test['Discuss_seq']

y_train = train.Score.values

batch_size = 32
epochs = 20
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=2)

callbacks_list = [early_stopping]
model = cnn()
model.summary()
# model.fit(X_train, y_train,validation_split=0.2,batch_size=batch_size, epochs=epochs, shuffle = True,callbacks=callbacks_list)

from sklearn.model_selection import KFold,StratifiedKFold
N = 4
kf = StratifiedKFold(n_splits=N, random_state=42)
# kf = KFold(n_splits=N,shuffle=False,random_state=42)
cv_pred = []
kf = kf.split(X_train,y_train)
xx_mse = []

# preds = model.predict(X_test)
# submission =pd.DataFrame(test.Id.values,columns=['Id'])
# submission['Score'] = preds
# submission.to_csv('cnn.csv',index=None,header =None)
from sklearn.metrics import mean_squared_error
def xx_mse_s(y_true,y_pre):
    y_true = pd.DataFrame({'res':list(y_true)})
    y_true['res'] = y_true['res'].astype(int)

    y_pre = pd.DataFrame({'res':list(y_pre)})
    y_pre['res'] = y_pre['res'].astype(int)

    return 1 / ( 1 + mean_squared_error(y_true['res'],y_pre['res'].values)**0.5)

for i ,(train_fold,test_fold) in enumerate(kf):
    X_train_1, X_validate, label_train, label_validate = X_train[train_fold, :], X_train[test_fold, :], y_train[train_fold], y_train[test_fold]

    file_path="weights_base.best.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    callbacks_list = [checkpoint, early] #early

    model.fit(X_train_1, label_train,epochs=20,batch_size=64,validation_data=(X_validate, label_validate),callbacks=callbacks_list,)

    # val_, result = fit_predict(X_train,label_train,X_validate,test)
    model.load_weights(file_path)
    val_ = model.predict(X_validate,batch_size=64)
    print(xx_mse_s(label_validate, val_))

    cv_pred.append(model.predict(X_test,batch_size=64))
    # cv_pred.append(result)
    xx_mse.append(xx_mse_s(label_validate, val_))

import numpy as np
print('xx_result',np.mean(xx_mse))

s = 0
for i in cv_pred:
    s = s + i

s = s/N
s = s[:,0]
print(s)
res = pd.DataFrame()
res['Id'] = list(test.Id.values)
res['pre'] = list(s)
# res['pre'] = res['pre'].astype(int)

res.to_csv('./20180301_cnn_1.csv',index=False,header=False)




