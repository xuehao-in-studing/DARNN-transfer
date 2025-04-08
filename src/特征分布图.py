import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据

src_data_path = "../data/ZJ/环切数据_DJ_5.csv"
tar_data_path = "../data/HZW/train/环切数据_DJ_5.csv"

# 读取数据
src_data = pd.read_csv(src_data_path)
tar_data = pd.read_csv(tar_data_path)

# 归一化
src_data_cols = src_data.columns[:]
tar_data_cols = tar_data.columns[:]
scaler_src = MinMaxScaler()
scaler_tar = MinMaxScaler()

src_data = scaler_src.fit_transform(src_data)
tar_data = scaler_tar.fit_transform(tar_data)
src_data = pd.DataFrame(src_data, columns=src_data_cols)
tar_data = pd.DataFrame(tar_data, columns=tar_data_cols)
src_data['domain'] = 'ZJ'
tar_data['domain'] = 'HZW'

view_cols = ["DJ_1", "DJ_2", "DJ_3", "DJ_4", "ZZJL", "ZXZ_YL_1", "KWC_YL_1"]
# 中文名
view_col_dict = {
    "DJ_1": "总推力",
    "DJ_2": "贯入度",
    "DJ_3": "刀盘转速",
    "DJ_4": "刀盘扭矩",
    "ZZJL": "总注浆量",
    "ZXZ_YL_1": "工作仓压力1",
    "KWC_YL_1": "开挖仓压力1",
}

# 合并数据并转换为长格式
df = pd.concat([src_data, tar_data], axis=0)
df = df.melt(id_vars=['domain'], value_vars=view_cols,
             var_name="variable", value_name="value")
# 将变量名称替换为中文
df['variable'] = df['variable'].replace(view_col_dict)
# 对数据进行排序

# # 输出两个不同域的刀盘转速和刀盘扭矩的统计特征
# describe = df.groupby(['domain', 'variable'])['value'].describe()
# print(describe)
# # df = df.sort_values(by=['domain', 'variable'])
# # 输出到文件
# describe.to_csv("../result/刀盘转速和刀盘扭矩的统计特征.csv", encoding='gbk')

print(df)
title_fontsize = 24
label_fontsize = 20

# 选择需要对比的特征
#
plt.figure(figsize=(16, 9), dpi=720)
sns.violinplot(data=df, x='variable', y='value', hue='domain')

plt.xticks(fontsize=label_fontsize, rotation=45)
plt.yticks([],fontsize=label_fontsize)
plt.xlabel("特征名称", fontsize=title_fontsize)
plt.ylabel("归一化后的值", fontsize=title_fontsize)
plt.tight_layout()
plt.legend(fontsize=label_fontsize)
plt.savefig("../plots/多特征分布对比.png", dpi=720)
plt.show()
