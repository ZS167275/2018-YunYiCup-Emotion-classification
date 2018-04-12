import pandas as pd
import numpy as np
np.random.seed(2018)
from keras import backend
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,TimeDistributed,Lambda ,CuDNNLSTM
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import word2vec
import gensim
import logging
from keras.preprocessing import text, sequence

save_model_name = '../data/mode.model'
print("reading data")
test_trans = pd.read_csv('../data/test_second_trans.csv')
train_and_test_trans = pd.read_csv('tmp.csv')
train = train_and_test_trans[~train_and_test_trans['Id'].isin(test_trans['Id'].unique())]
test = train_and_test_trans[train_and_test_trans['Id'].isin(test_trans['Id'].unique())]
print("set super paramter")

max_features = 50000
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

print("set label")
y = []
for i in list(train['Score']):
    y.append(i + 0.0)

# for i in list(train['Score']):
#     if i == 1 or i ==2 or i ==3:
#         y.append(i + np.random.random(1)* 0.5)
#     else:
#         y.append(i + np.random.random(1)* 1)

y = np.array(y).reshape(-1,1)[:,0]
train['Score'] = list(y)
print(train['Score'])
print(y)

print("finish rading")
train_row = train.shape[0]
df_train = train
df_test = test
test_id = test['Id']
print(test_id.values)
##word2vec模型训练
def model_train(train_file_name, save_model_file,n_dim):  # model_file_name为训练语料的路径,save_model为保存模型名
    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = gensim.models.Word2Vec(sentences, size=n_dim,window =8)  # 训练skip-gram模型; 默认window=5
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

# print('word2vec')
# train_and_test_trans['Trans'].to_csv('./cut.txt',index=False,header=False)
# model_train("./cut.txt", save_model_name,EMBEDDING_DIM)
# w2v_model = word2vec.Word2Vec.load(save_model_name)

X_train = train['Trans']
X_test = test['Trans']

# 创建词典，参数num_words 为词典的频次最大值
tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
y =df_train['Score'].values
print('Shape of test tensor:', test.shape)

##找出了lstm权重
word_index = tokenizer.word_index
# 该语料中单词的总数 max_features
nb_words = min(max_features, len(word_index))

print('Total %s word vectors.' % nb_words)

# 创建一个单词- 向量的词典
###计算权重
# embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i >= max_features:
#         continue
#     else :
#         try:
#             embedding_vector = w2v_model[word]
#         except KeyError:
#             continue
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, train_size=0.75, random_state=233)
print("x_tra shape ",X_tra.shape)
print("X_val shape ",X_val.shape)
print("y_tra shape ",y_tra.shape)

hidden_dim_1 = 200
hidden_dim_2 = 100
print('Build model...')
document = Input(shape = (None, ), dtype = "int32")
left_context = Input(shape = (None, ), dtype = "int32")
right_context = Input(shape = (None, ), dtype = "int32")

EMBEDDING_FILE2 =  './glove.840B.300d.txt'

emb_mean, emb_std = 0.0055286596, 0.34703913

embeddings_index_1 = {}
with open(EMBEDDING_FILE2,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index_1[word] = coefs

embedding_matrix_1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector_1 = embeddings_index_1.get(word)
    #embedding_vector_2 = embeddings_index_2.get(word)
    if embedding_vector_1 is not None: embedding_matrix_1[i] = embedding_vector_1
    #if embedding_vector_2 is not None: embedding_matrix_2[i] = embedding_vector_2


embedder = Embedding(max_features, 300, weights = [embedding_matrix_1], trainable = False)





doc_embedding = embedder(document)
l_embedding = embedder(left_context)
r_embedding = embedder(right_context)

# I use LSTM RNNs instead of vanilla RNNs as described in the paper.
forward = CuDNNLSTM(hidden_dim_1, return_sequences = True)(l_embedding) # See equation (1).
backward = CuDNNLSTM(hidden_dim_1, return_sequences = True, go_backwards = True)(r_embedding) # See equation (2).
together = concatenate([forward, doc_embedding, backward], axis = 2) # See equation (3).

semantic = TimeDistributed(Dense(hidden_dim_2, activation = "tanh"))(together) # See equation (4).

# Keras provides its own max-pooling layers, but they cannot handle variable length input
# (as far as I can tell). As a result, I define my own max-pooling layer here.
pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (hidden_dim_2, ))(semantic) # See equation (5).

output = Dense(1, input_dim = hidden_dim_2, activation = "linear")(pool_rnn) # See equations (6) and (7).NUM_CLASSES=1

model = Model(inputs = [document, left_context, right_context], outputs = output)
model.compile(optimizer = "adam", loss = "mse", metrics = ["mse"])


max_token = max_features
##生成左右上下文
print('Build left and right data')
doc_x_train = np.array(X_tra)
# We shift the document to the right to obtain the left-side contexts.
left_x_train = np.array([[max_token]+t_one[:-1].tolist() for t_one in X_tra])
# We shift the document to the left to obtain the right-side contexts.
right_x_train = np.array([t_one[1:].tolist()+[max_token] for t_one in X_tra])

x_test = X_val
doc_x_test = np.array(x_test)
# We shift the document to the right to obtain the left-side contexts.
left_x_test = np.array([[max_token]+t_one[:-1].tolist() for t_one in x_test])
# We shift the document to the left to obtain the right-side contexts.
right_x_test = np.array([t_one[1:].tolist()+[max_token] for t_one in x_test])

file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
callbacks_list = [checkpoint, early] #early

# history = model.fit([doc_x_train, left_x_train, right_x_train], y_train, epochs = 1)
# loss = history.history["loss"][0]
model.fit([doc_x_train, left_x_train, right_x_train], y_tra,
          batch_size=128,
          epochs=25,
          validation_data=[[doc_x_test, left_x_test, right_x_test], y_val],callbacks=callbacks_list)

model.load_weights(file_path)
# x_test = X_val
doc_x_test_1 = np.array(test)
# We shift the document to the right to obtain the left-side contexts.
left_x_test_1 = np.array([[max_token]+t_one[:-1].tolist() for t_one in test])
# We shift the document to the left to obtain the right-side contexts.
right_x_test_1 = np.array([t_one[1:].tolist()+[max_token] for t_one in test])

y_pred = model.predict([doc_x_test_1, left_x_test_1, right_x_test_1], batch_size=512)

y_pre_d = model.predict([doc_x_test, left_x_test, right_x_test], batch_size=512)



from sklearn.metrics import mean_squared_error
def xx_mse_s(y_true,y_pre):
    y_true = pd.DataFrame({'res':list(y_true)})
    y_true['res'] = y_true['res'].astype(int)

    y_pre = pd.DataFrame({'res':list(y_pre)})
    y_pre['res'] = y_pre['res'].astype(int)

    return 1 / ( 1 + mean_squared_error(y_true['res'],y_pre['res'].values)**0.5)

print(xx_mse_s(y_val,y_pre_d))

res = pd.DataFrame()
res['Id'] = list(test_id.values)
res['pre'] = list(y_pred[:,0])
# res['pre'] = res['pre'].astype(int)

res.to_csv('./2018-04-05-rcnn_w2v.csv',index=False,header=False)
