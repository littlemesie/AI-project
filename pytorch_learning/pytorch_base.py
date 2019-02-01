# -*- coding: utf-8 -*-
# 首先要引入相关的包
import torch
import numpy as np
#打印一下版本
print(torch.__version__)

x = torch.rand(2, 3)
print(x)
# 也可以使用size()函数，返回的结果都是相同的
print(x.size())
#我们直接使用现有数字生成
PI = torch.tensor(3.1415926)
print(PI)
# 对于标量，我们可以直接使用 .item() 从中取出其对应的python对象的数值
print(PI.item())

# tensor转化为numpy
numpy_x = x.numpy()
print(numpy_x)
torch_a = torch.from_numpy(numpy_x)
print(torch_a)


#使用torch.cuda.is_available()来确定是否有cuda设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 自动求导
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z = torch.sum(x+y)
print(z)
# 简单的自动求导
z.backward()
#看一下x和y的梯度
print(x.grad,y.grad)

# 复杂的求导
x1 = torch.rand(5, 5, requires_grad=True)
y1 = torch.rand(5, 5, requires_grad=True)
z1 = x1**2 + y1**3
print(z1)


#我们的返回值不是一个scalar，所以需要输入一个大小相同的张量作为参数，这里我们用ones_like函数根据x生成一个张量
z1.backward(torch.ones_like(x1))
print(x1.grad,y1.grad)