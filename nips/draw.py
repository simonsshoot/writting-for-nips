import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置学术风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Paired")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9
})

# 创建数据框
data = {
    'Type': ['Original', 'Original', 'Attacked', 'Attacked'],
    'Detectors': ['Single', 'Joint', 'Single', 'Joint'],
    'Human-recall': [89.48, 88.13, 86.90, 85.91],
    'Machine-recall': [92.25, 98.99, 94.13, 99.80],
    'Recall': [90.86, 93.56, 90.52, 92.86],
    'F1': [90.45, 93.15, 55.78, 90.39],
    'ACC': [91.44, 93.77, 93.81, 99.18]
}

df = pd.DataFrame(data).melt(id_vars=['Type', 'Detectors'], 
                           var_name='Metric',
                           value_name='Value')

# 创建三个子图布局
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)
plt.subplots_adjust(wspace=0.3)

# 子图1: Original场景
sns.barplot(ax=axes[0],
            x='Metric',
            y='Value',
            hue='Detectors',
            data=df[df['Type'] == 'Original'],
            edgecolor='black',
            linewidth=0.5,
            errwidth=0)
axes[0].set_title('Original Scenario', weight='bold', pad=12)
axes[0].set_ylim(50, 105)

# 子图2: Attacked场景
sns.barplot(ax=axes[1],
            x='Metric',
            y='Value',
            hue='Detectors',
            data=df[df['Type'] == 'Attacked'],
            edgecolor='black',
            linewidth=0.5,
            errwidth=0)
axes[1].set_title('Attacked Scenario', weight='bold', pad=12)
axes[1].set_ylim(50, 105)

# 子图3: 全局对比
sns.barplot(ax=axes[2],
            x='Metric',
            y='Value',
            hue='Type',
            data=df,
            edgecolor='black',
            linewidth=0.5,
            errwidth=0)
axes[2].set_title('Global Comparison', weight='bold', pad=12)
axes[2].set_ylim(50, 105)

# 统一设置公共元素
for ax in axes:
    # 添加数据标签
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}%', 
                    (p.get_x()+p.get_width()/2., height),
                    ha='center', va='bottom',
                    xytext=(0, 2),
                    textcoords='offset points',
                    fontsize=8)
    
    # 设置公共样式
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.8)
    ax.legend().set_visible(False)

# 添加全局图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, 
         title='Detectors',
         loc='upper center',
         bbox_to_anchor=(0.5, 0.98),
         ncol=2,
         frameon=True,
         shadow=True)

plt.tight_layout()
plt.show()