# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

# data = pd.read_csv('data_stocks.csv')
# plt.figure()
# x = [i for i,ind in enumerate(data['SP500'])]
# y = data['SP500']
# plt.plot(x,y)
# plt.show()

def get_data():
    """获取数据，切分成训练集和测试集"""
    data = pd.read_csv('data_stocks.csv')
    data.drop('DATE',axis=1, inplace=True)
    data_train = data.iloc[:int(data.shape[0] * 0.8), :]
    data_test = data.iloc[int(data.shape[0] * 0.8):, :]
    # 数据归一化到[-1,1]之间
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    return data_train,data_test

def tf_model(data_train,data_test):
    """使用 TensorFlow 实现同步预测"""
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    input_dim = X_train.shape[1]

    hidden_1 = 1024
    hidden_2 = 512
    hidden_3 = 256
    hidden_4 = 128
    output_dim = 1
    batch_size = 256
    epochs = 10

    tf.reset_default_graph()

    X = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
    Y = tf.placeholder(shape=[None], dtype=tf.float32)

    W1 = tf.get_variable('W1', [input_dim, hidden_1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [hidden_1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2', [hidden_1, hidden_2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [hidden_2], initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3', [hidden_2, hidden_3], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [hidden_3], initializer=tf.zeros_initializer())
    W4 = tf.get_variable('W4', [hidden_3, hidden_4], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable('b4', [hidden_4], initializer=tf.zeros_initializer())
    W5 = tf.get_variable('W5', [hidden_4, output_dim], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b5 = tf.get_variable('b5', [output_dim], initializer=tf.zeros_initializer())

    h1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))
    h3 = tf.nn.relu(tf.add(tf.matmul(h2, W3), b3))
    h4 = tf.nn.relu(tf.add(tf.matmul(h3, W4), b4))
    out = tf.transpose(tf.add(tf.matmul(h4, W5), b5))

    cost = tf.reduce_mean(tf.squared_difference(out, Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            shuffle_indices = np.random.permutation(np.arange(y_train.shape[0]))
            X_train = X_train[shuffle_indices]
            y_train = y_train[shuffle_indices]

            for i in range(y_train.shape[0] // batch_size):
                start = i * batch_size
                batch_x = X_train[start: start + batch_size]
                batch_y = y_train[start: start + batch_size]
                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

                if i % 50 == 0:
                    print('MSE Train:', sess.run(cost, feed_dict={X: X_train, Y: y_train}))
                    print('MSE Test:', sess.run(cost, feed_dict={X: X_test, Y: y_test}))
                    y_pred = sess.run(out, feed_dict={X: X_test})
                    y_pred = np.squeeze(y_pred)
                    plt.plot(y_test, label='test')
                    plt.plot(y_pred, label='pred')
                    plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                    plt.legend()
                    plt.show()

if __name__ == '__main__':
    data_train, data_test = get_data()
    tf_model(data_train, data_test)