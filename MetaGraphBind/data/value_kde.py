import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 Excel 文件（假设文件名为 "data.xlsx"，工作表为默认的第一个）
# df = pd.read_excel("R_7.18.xlsx")
df = pd.read_excel("R_5.22.xlsx")
# df = pd.read_excel("new_bro.xlsx")
# 确保列名正确
if 'Value' not in df.columns:
    raise ValueError("Excel 文件中没有名为 'Value' 的列")

# 直方图
fig = plt.figure(figsize=(6, 4))
title = fig.suptitle("Value Distribution", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
mean_val = df['Value'].mean()
ax.text(df['Value'].max()*0.8, df['Value'].count()/10,
        r'$\mu$='+str(round(mean_val, 2)), fontsize=12)

freq, bins, patches = ax.hist(df['Value'], color='steelblue', bins=15,
                              edgecolor='black', linewidth=1)

# 核密度图
# fig = plt.figure(figsize=(6, 4))
# title = fig.suptitle("Value Density Plot", fontsize=14)
# fig.subplots_adjust(top=0.85, wspace=0.3)
#
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.set_xlabel("Value")
# ax1.set_ylabel("Density")
# sns.kdeplot(df['Value'], ax=ax1, shade=True, color='steelblue')

plt.show()
