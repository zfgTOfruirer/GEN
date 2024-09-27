import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

# Fixing random state for reproducibility
#np.random.seed(19680801)

data = pd.read_csv(r'AB.csv',usecols=[0])
S1 = data.values #将数据赋给S1
S = S1[:,0]

dt = 0.01
t = np.arange(0, len(S), dt)
nse = np.random.randn(len(t))
r = np.exp(-t / 0.05)

cnse = np.convolve(nse, r) * dt
cnse = cnse[:len(t)]

s = S + cnse

fig, (ax0, ax1) = plt.subplots(2, 1)
ax0.plot(t, s)
ax1.psd(s, 256, 1 / dt)

plt.show()