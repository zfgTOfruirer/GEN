import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
#%%
np.random.seed(1018)
#%%
#日行驶里程s
s = np.random.lognormal(5.21, 0.34, 1000)
for i in range(1000):
    while s[i]>400:
        s[i] = np.random.lognormal(5.21, 0.34)
sns.displot(s)

plt.grid()
plt.savefig('./mc_ev_loadoA.png',dpi=400,bbox_inches = 'tight')
plt.xlabel("时刻/小时", fontdict={'weight': 'normal', 'size': 23})#改变坐标轴标题字体
plt.ylabel("功率/MW", fontdict={'weight': 'normal', 'size': 23})



#%%
#充电开始时间分布
# Number = 2000;   #样本个数
# mu = 20    #均值
# sigma = 1.33  #标准差
T1 = np.random.normal(9.5, 1, 1000)
T2 = np.random.normal(14, 1, 1000)
T3 = np.random.normal(19, 2, 1000)
sns.displot(T1)

plt.grid()
plt.savefig('./mc_ev_loadoB.png',dpi=400,bbox_inches = 'tight')



#%%
#创建二维数组，贴上标签：充电时间段1， 2， 3
T1_new = np.ones(1000)
T2_new = np.ones(1000)*2
T3_new = np.ones(1000)*3
T1_new = T1_new.astype('int')
T2_new = T2_new.astype('int')
T3_new = T3_new.astype('int')
T_1 = np.vstack((T1, T1_new))
T_2 = np.vstack((T2, T2_new))
T_3 = np.vstack((T3, T3_new))
''' #贴上全0标签，记录快充慢充
T4 = np.zeros(1000)
T4 = T4.astype('int')
T_1 = np.vstack((T_1, T4))
T_2 = np.vstack((T_2, T4))
T_3 = np.vstack((T_3, T4))
arr = np.zeros(1000).reshape(1, 1000)
arr[0,i] = 1
'''#%%
#选择上午、下午、晚上充电 对应概率0.2, 0.1, 0.7
def M_A_E():
    p =([0.2, 0.1, 0.7])
    value = np.random.choice([1, 2, 3], p = p)
    return value
#%%
#确定充电开始时间ts
ts = np.zeros((3, 1000))
for i in range(1000):
    value = M_A_E()
    if value == 1:
        ts[0][i] = T_1[0][i]
        ts[1][i] = T_1[1][i]
    elif value == 2:
        ts[0][i] = T_2[0][i]
        ts[1][i] = T_2[1][i]
    else:
        ts[0][i] = T_3[0][i]
        ts[1][i] = T_3[1][i]
#     print(value)
#     print(ts[1][i])
ts[0,:]*=3600
ts= ts.astype(np.int32)
# ts
#%%
s_max = 400        #最大里程，单位km
soc0 = (1-s/s_max) #初始SOC0
fc = 60            #快速充电60kw
cc = 7             #常规充电7kw
# soc0.min()
#%%
#选择充电功率c
for i in range(1000):
    if ((ts[1][i] == 1) and (soc0[i]<=0.8)):
        ts[2][i] = fc
    elif ((ts[1][i] == 2) and (soc0[i]<=0.8)):
        ts[2][i] = fc
    elif ((ts[1][i] == 3) and (soc0[i]<=0.8)):
        ts[2][i] = cc
#     else:
#         continue
#%%
#确定充电持续时间tc
tc = np.zeros(1000)
for i in range(1000):
    if(ts[2][i]==0):
        tc[i] = 0
    else:
        a = (1-soc0[i])*82/(0.9*ts[2][i])
        b = s[i]*20.5/(100*0.9*ts[2][i])
        tc[i] = np.minimum(a, b)
tc *= 3600
tc= tc.astype(np.int32)
#tc
#%%
#确定充电结束时间te
te = np.zeros(1000)
for i in range(1000):
    te[i] = ts[0][i] + tc[i]
te= te.astype(np.int32)
#te.max()
#%%
Time_P = np.zeros((86400*2,1000)) #时间-功率二维数组
for i in range(1000):
    row_s = ts[0][i]
    row_e = te[i]
    Time_P[row_s:row_e+1, i] = ts[2][i]
#%%
x = np.arange(86400*2)/3600
Y = np.array(Time_P)





y = Y.sum(axis=1)/1000

Frame = pd.DataFrame(x,y)
Frame.to_csv('CARsamples.csv', header=None, index=None)

plt.figure(figsize=(22, 10))
plt.rcParams['font.sans-serif']=['SimHei']###解决中文乱码
plt.rcParams['axes.unicode_minus']=False
plt.plot(x,y,'b')
plt.xticks(range(0,49))
plt.xlim(0,48)
plt.xlabel("时刻/小时", fontdict={'weight': 'normal', 'size': 23})#改变坐标轴标题字体
plt.ylabel("功率/MW", fontdict={'weight': 'normal', 'size': 23})
plt.tick_params(labelsize=20)
plt.grid()
plt.savefig('./mc_ev_loadone.png',dpi=400,bbox_inches = 'tight')
plt.plot(x, y)
#%%
#第一辆车功率图

x = np.arange(86400*2)/3600
y = Time_P[:, 0] #第一辆车功率变化


plt.figure(figsize=(22,10))
plt.rcParams['font.sans-serif']=['SimHei']###解决中文乱码
plt.rcParams['axes.unicode_minus']=False
plt.plot(x,y,'b')
plt.xticks(range(0,49))
plt.xlim(0,48)
plt.xlabel("时刻/小时", fontdict={'weight': 'normal', 'size': 23})#改变坐标轴标题字体
plt.ylabel("功率/KW", fontdict={'weight': 'normal', 'size': 23})
plt.tick_params(labelsize=20)

plt.grid()
plt.savefig('./mc_ev_loadtwo.png',dpi=400,bbox_inches = 'tight')
plt.plot(x, y)
