# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt  # 可视化
import seaborn as sns  # 可视化

df = pd.read_csv('prepareedcarlod.csv')  # 读取数据

print('pearson\n', df.corr(method='pearson'))  # 皮尔逊相关系数

#print('kendall\n', df.corr(method='kendall'))  # 肯德尔秩相关系数
# print('spearman\n', df.corr(method='spearman'))  # 斯皮尔曼秩相关系数

_, ax = plt.subplots(figsize=(12, 10))  # 分辨率1200×1000
corr = df.corr(method='pearson')  # 使用皮尔逊系数计算列与列的相关性
#corr = df.corr(method='kendall')

# corr = df.corr(method='spearman')
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 在两种HUSL颜色之间制作不同的调色板。图的正负色彩范围为220、10，结果为真则返回matplotlib的colormap对象
_ = sns.heatmap(
    corr,  # 使用Pandas DataFrame数据，索引/列信息用于标记列和行
    cmap=cmap,  # 数据值到颜色空间的映射
    square=True,  # 每个单元格都是正方形
    cbar_kws={'shrink': .9},  # `fig.colorbar`的关键字参数
    ax=ax,  # 绘制图的轴
    annot=True,  # 在单元格中标注数据值
    annot_kws={'fontsize': 12})  # 热图，将矩形数据绘制为颜色编码矩阵

plt.show()


