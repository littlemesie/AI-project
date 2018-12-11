# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/11 22:48
"""
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, TimeDistributed, Bidirectional
from keras.models import Model, load_model
from keras.utils import np_utils
import numpy as np
import re

# 读取字典
vocab = open('data/msr/msr_training_words.utf8').read().rstrip('\n').split('\n')
vocab = list(''.join(vocab))
stat = {}
for v in vocab:
    stat[v] = stat.get(v, 0) + 1
stat = sorted(stat.items(), key=lambda x:x[1], reverse=True)
vocab = [s[0] for s in stat]
# 5167 个字
print(len(vocab))
# 映射
char2id = {c : i + 1 for i, c in enumerate(vocab)}
id2char = {i + 1 : c for i, c in enumerate(vocab)}
tags = {'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4}

embedding_size = 128
maxlen = 32 # 长于32则截断，短于32则填充0
hidden_size = 64
batch_size = 64
epochs = 50


def load_data(path):
    data = open(path).read().rstrip('\n')
    # 按标点符号和换行符分隔
    data = re.split('[，。！？、\n]', data)
    print('共有数据 %d 条' % len(data))
    print('平均长度：', np.mean([len(d.replace(' ', '')) for d in data]))

    # 准备数据
    X_data = []
    y_data = []

    for sentence in data:
        sentence = sentence.split(' ')
        X = []
        y = []

        try:
            for s in sentence:
                s = s.strip()
                # 跳过空字符
                if len(s) == 0:
                    continue
                # s
                elif len(s) == 1:
                    X.append(char2id[s])
                    y.append(tags['s'])
                elif len(s) > 1:
                    # b
                    X.append(char2id[s[0]])
                    y.append(tags['b'])
                    # m
                    for i in range(1, len(s) - 1):
                        X.append(char2id[s[i]])
                        y.append(tags['m'])
                    # e
                    X.append(char2id[s[-1]])
                    y.append(tags['e'])

            # 统一长度
            if len(X) > maxlen:
                X = X[:maxlen]
                y = y[:maxlen]
            else:
                for i in range(maxlen - len(X)):
                    X.append(0)
                    y.append(tags['x'])
        except:
            continue
        else:
            if len(X) > 0:
                X_data.append(X)
                y_data.append(y)

    X_data = np.array(X_data)
    y_data = np_utils.to_categorical(y_data, 5)

    return X_data, y_data


X_train, y_train = load_data('data/msr/msr_training.utf8')
X_test, y_test = load_data('data/msr/msr_test_gold.utf8')
print('X_train size:', X_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', X_test.shape)
print('y_test size:', y_test.shape)


X = Input(shape=(maxlen,), dtype='int32')
embedding = Embedding(input_dim=len(vocab) + 1, output_dim=embedding_size, input_length=maxlen, mask_zero=True)(X)
blstm = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(embedding)
blstm = Dropout(0.6)(blstm)
blstm = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(blstm)
blstm = Dropout(0.6)(blstm)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)

model = Model(X, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
model.save('msr_bilstm.h5')


def viterbi(nodes):
    trans = {'be': 0.5, 'bm': 0.5, 'eb': 0.5, 'es': 0.5, 'me': 0.5, 'mm': 0.5, 'sb': 0.5, 'ss': 0.5}
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}
    for l in range(1, len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():
            nows = {}
            for j in paths_.keys():
                if j[-1] + i in trans.keys():
                    nows[j + i] = paths_[j] + nodes[l][i] + trans[j[-1] + i]
            nows = sorted(nows.items(), key=lambda x: x[1], reverse=True)
            paths[nows[0][0]] = nows[0][1]

    paths = sorted(paths.items(), key=lambda x: x[1], reverse=True)
    return paths[0][0]


def cut_words(data):
    data = re.split('[，。！？、\n]', data)
    sens = []
    Xs = []
    for sentence in data:
        sen = []
        X = []
        sentence = list(sentence)
        for s in sentence:
            s = s.strip()
            if not s == '' and s in char2id:
                sen.append(s)
                X.append(char2id[s])
        if len(X) > maxlen:
            sen = sen[:maxlen]
            X = X[:maxlen]
        else:
            for i in range(maxlen - len(X)):
                X.append(0)

        if len(sen) > 0:
            Xs.append(X)
            sens.append(sen)

    Xs = np.array(Xs)
    ys = model.predict(Xs)

    results = ''
    for i in range(ys.shape[0]):
        nodes = [dict(zip(['s', 'b', 'm', 'e'], d[:4])) for d in ys[i]]
        ts = viterbi(nodes)
        for x in range(len(sens[i])):
            if ts[x] in ['s', 'e']:
                results += sens[i][x] + '/'
            else:
                results += sens[i][x]

    return results[:-1]