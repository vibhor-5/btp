import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_metrics():
    # Data derived from authentic DataCo baseline run
    models = ['XGBoost', 'Random Forest']
    
    # Metrics
    roc_auc = [0.8728, 0.8724]
    precision = [0.69, 0.69]
    recall = [0.98, 0.79]
    f1_score = [0.81, 0.73]
    
    x = np.arange(len(models))
    width = 0.2
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - 1.5*width, roc_auc, width, label='ROC-AUC', color='#4c72b0')
    rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#dd8452')
    rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#55a868')
    rects4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', color='#c44e52')
    
    ax.set_ylabel('Scores')
    ax.set_title('Baseline Model Performance Comparison on Authentic DataCo Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    
    ax.set_ylim(0, 1.1)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    fig.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300)
    print("Saved comparison chart to 'metrics_comparison.png'")

if __name__ == "__main__":
    plot_metrics()
