import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as pt

# 在0-2*pi的区间上生成100个点作为输入数据

#Y = np.sin(X)

data = pd.read_csv(r'loadA.csv',usecols=[0]) #usecols=[0,1,2] 表述读取第1，2，3列
S1 = data.values #将数据赋给S1
S = S1[:,0]
t = np.arange(0,len(S),1)

#t = np.linspace(0,2*np.pi,100,endpoint=True)


# 对输入数据加入gauss噪声
# 定义gauss噪声的均值和方差
mu = 0
sigma =0.001#sigma = 0.04 #

for i in range(t.size):
    t[i] += random.gauss(mu,sigma)
    S[i] += random.gauss(mu,sigma)

# 画出这些点
pt.plot(t,S,linestyle='',marker='.')
pt.show()
Frame = pd.DataFrame(S)
Frame.to_csv('loadZAO.csv', header=None, index=None)