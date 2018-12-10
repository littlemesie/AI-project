# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import re
import time

# 读取字典
vocab = open('data/msr/msr_training_words.utf8').read().rstrip('\n').split('\n')
vocab = list(''.join(vocab))
stat = {}
for v in vocab:
    # 统计出现的次数
    stat[v] = stat.get(v, 0) + 1
# 按照降序排列
stat = sorted(stat.items(), key=lambda x: x[1], reverse=True)
vocab = [s[0] for s in stat]
# 5167 个字
print(len(vocab))
# 映射
char2id = {c: i + 1 for i, c in enumerate(vocab)}
# key-value 交换
id2char = {i + 1: c for i, c in enumerate(vocab)}
tags = {'s': [1, 0, 0, 0], 'b': [0, 1, 0, 0], 'm': [0, 0, 1, 0], 'e': [0, 0, 0, 1]}
print(tags)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph('./msr_fcn/msr_fcn.meta')
saver.restore(sess, tf.train.latest_checkpoint('./msr_fcn'))

graph = tf.get_default_graph()
X_input = graph.get_tensor_by_name('X_input:0')
Y_pred = graph.get_tensor_by_name('Y_pred:0')


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

        if len(X) > 0:
            Xs.append(X)
            sens.append(sen)

    results = ''
    for i in range(len(Xs)):
        X_d = np.array([Xs[i]])
        print(X_d)
        Y_d = sess.run(Y_pred, feed_dict={X_input: X_d})
        print(Y_d)
        nodes = [dict(zip(['s', 'b', 'm', 'e'], d)) for d in Y_d[0]]

        ts = viterbi(nodes)
        for x in range(len(sens[i])):
            if ts[x] in ['s', 'e']:
                results += sens[i][x] + '/'
            else:
                results += sens[i][x]

    return results[:-1]


# print(cut_words('中国共产党第十九次全国代表大会，是在全面建成小康社会决胜阶段、中国特色社会主义进入新时代的关键时期召开的一次十分重要的大会。'))
# print(cut_words('把这本书推荐给，具有一定编程基础，希望了解数据分析、人工智能等知识领域，进一步提升个人技术能力的社会各界人士。'))
# print(cut_words('结婚的和尚未结婚的。'))
print(cut_words('我来到北京清华大学'))