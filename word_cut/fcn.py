# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/9 22:21
"""

import tensorflow as tf
import numpy as np
import re
import time

# 读取字典
vocab = open('data/msr/msr_training_words.utf8').read().rstrip('\n').split('\n')
vocab = list(''.join(vocab))
stat = {}
for v in vocab:
    stat[v] = stat.get(v, 0) + 1
stat = sorted(stat.items(), key=lambda x: x[1], reverse=True)
vocab = [s[0] for s in stat]
# 5167 个字
print(len(vocab))
# 映射
char2id = {c: i + 1 for i, c in enumerate(vocab)}
id2char = {i + 1: c for i, c in enumerate(vocab)}
tags = {'s': [1, 0, 0, 0], 'b': [0, 1, 0, 0], 'm': [0, 0, 1, 0], 'e': [0, 0, 0, 1]}
batch_size = 64

def get_dic():
    # 读取字典
    vocab = open('data/msr/msr_training_words.utf8').read().rstrip('\n').split('\n')
    vocab = list(''.join(vocab))
    stat = {}
    for v in vocab:
        stat[v] = stat.get(v, 0) + 1
    stat = sorted(stat.items(), key=lambda x: x[1], reverse=True)
    vocab = [s[0] for s in stat]
    # 5167 个字
    print(len(vocab))
    # 映射
    char2id = {c: i + 1 for i, c in enumerate(vocab)}
    id2char = {i + 1: c for i, c in enumerate(vocab)}

    return char2id, id2char

def load_data(path):
    data = open(path).read().rstrip('\n')
    # 按标点符号和换行符分隔
    data = re.split('[，。！？、\n]', data)
    # 准备数据
    X_data = []
    Y_data = []
    for sentence in data:
        sentence = sentence.split(' ')
        X = []
        Y = []

        try:
            for s in sentence:
                s = s.strip()
                # 跳过空字符
                if len(s) == 0:
                    continue
                # s
                elif len(s) == 1:
                    X.append(char2id[s])
                    Y.append(tags['s'])
                elif len(s) > 1:
                    # b
                    X.append(char2id[s[0]])
                    Y.append(tags['b'])
                    # m
                    for i in range(1, len(s) - 1):
                        X.append(char2id[s[i]])
                        Y.append(tags['m'])
                    # e
                    X.append(char2id[s[-1]])
                    Y.append(tags['e'])
        except:
            continue
        else:
            if len(X) > 0:
                X_data.append(X)
                Y_data.append(Y)

    order = np.argsort([len(X) for X in X_data])
    X_data = [X_data[i] for i in order]
    Y_data = [Y_data[i] for i in order]

    current_length = len(X_data[0])
    X_batch = []
    Y_batch = []
    for i in range(len(X_data)):
        if len(X_data[i]) != current_length or len(X_batch) == batch_size:
            yield np.array(X_batch), np.array(Y_batch)

            current_length = len(X_data[i])
            X_batch = []
            Y_batch = []

        X_batch.append(X_data[i])
        Y_batch.append(Y_data[i])

embedding_size = 128

embeddings = tf.Variable(tf.random_uniform([len(char2id) + 1, embedding_size], -1.0, 1.0))
X_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='X_input')
embedded = tf.nn.embedding_lookup(embeddings, X_input)

W_conv1 = tf.Variable(tf.random_uniform([3, embedding_size, embedding_size // 2], -1.0, 1.0))
b_conv1 = tf.Variable(tf.random_uniform([embedding_size // 2], -1.0, 1.0))
Y_conv1 = tf.nn.relu(tf.nn.conv1d(embedded, W_conv1, stride=1, padding='SAME') + b_conv1)

W_conv2 = tf.Variable(tf.random_uniform([3, embedding_size // 2, embedding_size // 4], -1.0, 1.0))
b_conv2 = tf.Variable(tf.random_uniform([embedding_size // 4], -1.0, 1.0))
Y_conv2 = tf.nn.relu(tf.nn.conv1d(Y_conv1, W_conv2, stride=1, padding='SAME') + b_conv2)

W_conv3 = tf.Variable(tf.random_uniform([3, embedding_size // 4, 4], -1.0, 1.0))
b_conv3 = tf.Variable(tf.random_uniform([4], -1.0, 1.0))
Y_pred = tf.nn.softmax(tf.nn.conv1d(Y_conv2, W_conv3, stride=1, padding='SAME') + b_conv3, name='Y_pred')

Y_true = tf.placeholder(dtype=tf.float32, shape=[None, None, 4], name='Y_true')
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_true * tf.log(Y_pred + 1e-20), axis=[2]))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y_pred, 2), tf.argmax(Y_true, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
max_test_acc = -np.inf

epochs = 50
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    train = load_data('data/msr/msr_training.utf8')
    accs = []
    i = 0
    t0 = int(time.time())
    for X_batch, Y_batch in train:
        sess.run(optimizer, feed_dict={X_input: X_batch, Y_true: Y_batch})
        i += 1
        if i % 100 == 0:
            acc = sess.run(accuracy, feed_dict={X_input: X_batch, Y_true: Y_batch})
            accs.append(acc)
    print('Epoch %d time %ds' % (e + 1, int(time.time()) - t0))
    print('- train accuracy: %f' % (np.mean(accs)))

    test = load_data('data/msr/msr_test_gold.utf8')
    accs = []
    for X_batch, Y_batch in test:
        acc = sess.run(accuracy, feed_dict={X_input: X_batch, Y_true: Y_batch})
        accs.append(acc)
    mean_test_acc = np.mean(accs)
    print('- test accuracy: %f' % mean_test_acc)

    if mean_test_acc > max_test_acc:
        max_test_acc = mean_test_acc
        print('Saving Model......')
        saver.save(sess, './msr_fcn/msr_fcn')
#
# if __name__ == '__main__':
#
#     char2id, id2char = get_dic()