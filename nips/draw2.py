import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置学术风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams.update({
    'font.family': 'Arial',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.title_fontsize': 12
})

# 创建数据框
data = {
    'Type': ['Original']*3 + ['Attacked']*3,
    'Detectors': ['First', 'Second', 'Combined']*2,
    'H-Rec': [88.75, 81.10, 87.72, 68.54, 77.00, 92.19],  # Human-recall
    'M-Rec': [92.30, 95.23, 98.94, 98.32, 99.85, 99.98],  # Machine-recall
    'Rec':  [90.52, 88.16, 93.33, 83.43, 88.42, 96.08],  # Recall
    'F1': [90.07, 87.09, 92.89, 74.95, 85.51, 95.71],  # F1
    'ACC':[90.60, 88.40, 93.55, 96.94, 97.83, 99.63]   # ACC
}

df = pd.DataFrame(data).melt(id_vars=['Type', 'Detectors'], 
                           var_name='Metric',
                           value_name='Value')

# 创建三个子图
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)
plt.subplots_adjust(wspace=0.25, top=0.85)

# 绘制每个检测器的子图
detectors = ['First', 'Second', 'Combined']
metric_order = ['H-Rec', 'M-Rec', 'Rec', 'F1', 'ACC']

# 子图标题颜色
sub_title_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色, 橙色, 绿色

for idx, detector in enumerate(detectors):
    ax = axes[idx]
    subset = df[df['Detectors'] == detector]
    
    sns.barplot(ax=ax,
                x='Metric',
                y='Value',
                hue='Type',
                data=subset,
                order=metric_order,
                edgecolor='black',
                linewidth=0.8,
                errwidth=0)
    
    # 子图装饰
    ax.set_title(f'{detector} Model', fontweight='bold', pad=12, color=sub_title_colors[idx])  # 设置子图标题颜色
    ax.set_ylim(60, 105)
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Score (%)' if idx == 0 else '', fontweight='bold', labelpad=-5)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 添加数据标签
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}',
                    (p.get_x() + p.get_width()/2, height),
                    ha='center', va='bottom',
                    xytext=(0, 3),
                    textcoords='offset points',
                    fontsize=9)
    
    # 移除重复图例
    if idx != 0:
        ax.get_legend().remove()
    else:
        ax.legend().set_title('Scenario')

# 添加全局标题
fig.suptitle("Performance Comparison Across Different Scenarios", 
             y=0.98, 
             fontsize=16, 
             fontweight='bold', 
             color='#333333')  # 全局标题颜色设置为深灰色

plt.tight_layout()
plt.show()
