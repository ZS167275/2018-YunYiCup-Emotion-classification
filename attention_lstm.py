import os
import re
import csv
import codecs
import pandas as pd 
import numpy as np
np.random.seed(2018)
import re
import jieba
from string import punctuation
from gensim.models import KeyedVectors
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import word2vec
import gensim
import logging
from keras.preprocessing import text, sequence
from keras.models import Sequential
import pickle




import sys

########################################
## set directories and parameters
########################################



from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim



train_name = "../data/train_first.csv"
predict_name = "../data/predict_first.csv"

cut_txt = '../data/dl_text.txt'  # 须注意文件必须先另存为utf-8编码格式
save_model_name = '../data/Word300.model'
save_feature = '../data/df_fea.pkl'
###w2v的特征维度
max_features = 30000
maxlen = 200
validation_split = 0.1

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 30000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

embed_size = 300

num_lstm = 300
num_dense = 256
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

act = 'relu'

train = pd.read_csv(train_name)
train_row = train.shape[0]
df_train = train
predict = pd.read_csv(predict_name)
df_test = predict

##word2vec模型训练
def model_train(train_file_name, save_model_file,n_dim):  # model_file_name为训练语料的路径,save_model为保存模型名
    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = gensim.models.Word2Vec(sentences, size=n_dim)  # 训练skip-gram模型; 默认window=5
    model.save(save_model_file)
#     model.wv.save_word2vec_format(save_model_name + ".bin", binary=True)   # 以二进制类型保存模型以便重用###转换词向量

###对词向量的转换
def gen_vec(text):
    vec = np.zeros(embed_size).reshape((1, embed_size))
    count = 0
    for word in text:
        try:
            vec += w2v_model[word].reshape((1, embed_size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec



if not os.path.exists(save_model_name):     # 判断文件是否存在
    model_train(cut_txt, save_model_name,EMBEDDING_DIM)
else:
    print('此训练模型已经存在，不用再次训练')
##中文分词
w2v_model = word2vec.Word2Vec.load(save_model_name)

###持久化操作，方便读取
if not os.path.exists(save_feature):
    df = pd.concat([df_train,df_test])
    df['cut_discuss'] = df['Discuss'].map(lambda x : " ".join(i for i in jieba.cut(x)))
    fw = open(save_feature,'wb') 
    pickle.dump(df,fw)
    fw.close()
else:
    print("特征存在，直接加载...")
    fw = open(save_feature,'rb') 
    df = pickle.load(fw)
    fw.close()

###第三列即是‘cut_discuss’,也就是取出第三列的所有行
X_train = df.iloc[:train_row,3]
X_test = df.iloc[train_row:,3]

tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
# X_train = np.array(X_train)
test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
# X_test = np.mat(X_test)
y =df_train['Score'].values
# print('Shape of data tensor:', X_train.shape)
# print('Shape of label tensor:', y.shape)
print('Shape of test tensor:', test.shape)

##找出了lstm权重
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

print('Total %s word vectors.' % nb_words)

###计算权重
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= max_features: 
        continue
    else :
        try:
            embedding_vector = w2v_model[word]
        except KeyError:
            continue
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector



########################################
## sample train/validation data
########################################
# np.random.seed(1234)
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, train_size=0.85, random_state=233)
print("x_tra shape ",X_tra.shape)
print("X_val shape ",X_val.shape)
print("y_tra shape ",y_tra.shape)

########################################
## define the model structure
########################################
###keras的model函数式，copy from  kaggle
def get_model():
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)

    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True)

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences= embedding_layer(inp)
    x = lstm_layer(embedded_sequences)
    x = Dropout(rate_drop_dense)(x)
    merged = Attention(MAX_SEQUENCE_LENGTH)(x)
    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    outp = Dense(1)(merged)

    ###这里的参数可以修正
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['mse'])

    return model

model = get_model()
print(model.summary())

STAMP = 'simple_lstm_w2v_vectors_%.2f_%.2f'%(rate_drop_lstm,rate_drop_dense)
print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=2)
bst_model_path = "../data/" + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

batch_size = 64
epochs = 20

hist = model.fit(X_tra, y_tra, \
        validation_data=(X_val, y_val), \
        epochs=epochs, batch_size=batch_size, shuffle=True, \
         callbacks=[early_stopping,model_checkpoint])
         
model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])

# #######################################
# ## make the submission
# ########################################
print('Start making the submission before fine-tuning')

y_test = model.predict(test, batch_size=256, verbose=1)

sample_submission = pd.read_csv("../data/submite_sample.csv",names = ['id','y'])
sample_submission['y'] = y_test

sample_submission.to_csv("../result/"+'%.4f_'%(bst_val_score)+STAMP+'.csv', index=False,encoding='utf-8',header=False)
print("It all is ok")
