# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2018/12/9 21:28
@summary:
"""
a = "你好北京"
a = list(a)
print(a)

nodes = [dict(zip(['s', 'b', 'm', 'e'], d)) for d in a]
print(nodes)