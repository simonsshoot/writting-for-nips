import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置专业级可视化样式
sns.set_style("whitegrid", {
    'grid.linestyle': ':',
    'grid.alpha': 0.4,
    'axes.edgecolor': '0.15',
    'axes.linewidth': 1.2
})
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# 定义专业配色方案
COLOR_SCHEME = {
    'contrastive': '#4285F4',  # Google Blue
    'crossentropy': '#EA4335',  # Google Red
    'text': '#5F6368'          # Google Gray
}

# 加载并预处理数据
def load_process_data():
    # 数据加载（请确认文件路径）
    acc_contrastive = pd.read_csv(r"C:\Users\admin\Desktop\process\smooth_train_acc_oneloss.csv")
    acc_crossentropy = pd.read_csv(r"C:\Users\admin\Desktop\process\smooth_train_acc_onlyclass.csv")
    loss_contrastive = pd.read_csv(r"C:\Users\admin\Desktop\process\smooth_loss_oneloss.csv")
    loss_crossentropy = pd.read_csv(r"C:\Users\admin\Desktop\process\smooth_loss_onlyclass.csv")
    
    # 数据合并
    acc_df = pd.merge(acc_contrastive, acc_crossentropy, on="Step", 
                     suffixes=('_Contrastive', '_CrossEntropy'))
    loss_df = pd.merge(loss_contrastive, loss_crossentropy, on="Step", 
                      suffixes=('_Contrastive', '_CrossEntropy'))
    return acc_df, loss_df

# 智能标注系统
def smart_annotator(ax, df, col, color, pos_type, label_position):
    """专业级标注工具，支持动态位置调整"""
    # 定位极值点
    idx = df[col].idxmax() if pos_type == 'max' else df[col].idxmin()
    x_val = df.loc[idx, 'Step']
    y_val = df.loc[idx, col]
    
    # 动态偏移参数
    offset_config = {
        'acc_contrastive': (-18, 'top'),
        'acc_crossentropy': (18, 'bottom'),
        'loss_both': (-18, 'top')
    }
    
    vertical_offset, va = offset_config[label_position]
    
    # 专业标注样式
    ax.annotate(f'{y_val:.3f}',
                xy=(x_val, y_val),
                xytext=(0, vertical_offset),
                textcoords='offset points',
                ha='center', 
                va=va,
                color=color,
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', 
                          fc='white', ec=color, lw=1.2, alpha=0.95),
                arrowprops=dict(arrowstyle='->', 
                                color=color,
                                linewidth=1.2,
                                connectionstyle="arc3,rad=0.25"))

# 主可视化函数
def create_professional_plots(acc_df, loss_df):
    # 初始化画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.subplots_adjust(top=0.92, wspace=0.28)
    
    # ================== 准确率可视化 ==================
    # 绘制主曲线
    ax1.plot(acc_df['Step'], acc_df['Value_Contrastive'], 
            label='Contrastive', linewidth=2.4, color=COLOR_SCHEME['contrastive'], zorder=3)
    ax1.plot(acc_df['Step'], acc_df['Value_CrossEntropy'], 
            label='Cross-Entropy', linewidth=2.4, color=COLOR_SCHEME['crossentropy'], zorder=3)
    
    # 添加采样标记
    mark_interval = len(acc_df) // 10
    for method in ['Contrastive', 'CrossEntropy']:
        ax1.scatter(acc_df['Step'][::mark_interval], 
                   acc_df[f'Value_{method}'][::mark_interval],
                   s=72, edgecolor='white', linewidth=1.4,
                   facecolor=COLOR_SCHEME['contrastive' if 'Contrastive' in method else 'crossentropy'],
                   marker='o' if 'Contrastive' in method else 's', zorder=4)
    
    # 添加智能标注
    smart_annotator(ax1, acc_df, 'Value_Contrastive', COLOR_SCHEME['contrastive'], 'max', 'acc_contrastive')
    smart_annotator(ax1, acc_df, 'Value_CrossEntropy', COLOR_SCHEME['crossentropy'], 'max', 'acc_crossentropy')
    
    # ================== 损失值可视化 ==================
    # 绘制主曲线
    ax2.plot(loss_df['Step'], loss_df['Value_Contrastive'], 
            label='Contrastive', linewidth=2.4, color=COLOR_SCHEME['contrastive'], zorder=3)
    ax2.plot(loss_df['Step'], loss_df['Value_CrossEntropy'], 
            label='Cross-Entropy', linewidth=2.4, color=COLOR_SCHEME['crossentropy'], zorder=3)
    
    # 添加采样标记
    for method in ['Contrastive', 'CrossEntropy']:
        ax2.scatter(loss_df['Step'][::mark_interval], 
                   loss_df[f'Value_{method}'][::mark_interval],
                   s=72, edgecolor='white', linewidth=1.4,
                   facecolor=COLOR_SCHEME['contrastive' if 'Contrastive' in method else 'crossentropy'],
                   marker='o' if 'Contrastive' in method else 's', zorder=4)
    
    # 添加智能标注
    smart_annotator(ax2, loss_df, 'Value_Contrastive', COLOR_SCHEME['contrastive'], 'min', 'loss_both')
    smart_annotator(ax2, loss_df, 'Value_CrossEntropy', COLOR_SCHEME['crossentropy'], 'min', 'loss_both')
    
    # ================== 专业样式配置 ==================
    for ax, title in zip([ax1, ax2], ['Training Accuracy', 'Training Loss']):
        # 标题和坐标轴
        ax.set_title(f'{title} Comparison', pad=18, color=COLOR_SCHEME['text'], fontweight='semibold')
        ax.set_xlabel('Training Steps', labelpad=10, color=COLOR_SCHEME['text'])
        ax.set_ylabel(title.split()[-1], labelpad=12, color=COLOR_SCHEME['text'])
        
        # 坐标轴美化
        ax.tick_params(axis='both', which='major', length=7, width=1.4, 
                      labelsize=11, colors=COLOR_SCHEME['text'])
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(1.4)
        
        # 动态调整坐标范围
        data_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.set_ylim(ax.get_ylim()[0] - data_range*0.07, 
                   ax.get_ylim()[1] + data_range*0.07)
        
        # 专业图例
        ax.legend(frameon=True, framealpha=0.95, 
                 loc='lower right' if 'Accuracy' in title else 'upper right',
                 fontsize=11, handlelength=1.5)

    # 输出专业级图像
    plt.tight_layout()
    plt.savefig('professional_training_comparison.png', dpi=400, 
               bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

# 执行主程序
if __name__ == "__main__":
    acc_data, loss_data = load_process_data()
    create_professional_plots(acc_data, loss_data)