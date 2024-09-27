
# 导入工具库
import pandas as pd
import numpy as np
from PyEMD import EMD, Visualisation


data = pd.read_csv(r'xingload.csv',usecols=[0]) #usecols=[0,1,2] 表述读取第1，2，3列
S1 = data.values #将数据赋给S1

S = S1[:,0]



# 构建信号
#t = np.arange(0,1, 0.01)
#S = 2*np.sin(2*np.pi*15*t) +4*np.sin(2*np.pi*10*t)*np.sin(2*np.pi*t*0.1)+np.sin(2*np.pi*5*t)
t = np.arange(0,len(S),1)
# 提取imfs和剩余信号res
emd = EMD()
emd.emd(S)
imfs, res = emd.get_imfs_and_residue()
# 绘制 IMF
vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)

# 绘制并显示所有提供的IMF的瞬时频率
vis.plot_instant_freq(t, imfs=imfs)
vis.show()
