import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import pandas as pd

def create_heatmap(label, predictions, idx):
    """混同行列をヒートマップで表す

    Args:
        label (list) : テストデータのlabelのlist
        predictions (list) : テストデータの予測のlist
        idx (int) : テストデータにするインデックス

    return:
        None
    """

    accuracy_nums = confusion_matrix(label, predictions)

    accuracy_sum = np.sum(accuracy_nums, axis=1)
    accuracy_rate = [accuracy_nums[idx] / accuracy_sum[idx] for idx in range(49)]

    accuracy_rate = np.round(accuracy_rate, decimals=2)
        
    with open('train_setting.yml', 'r') as yml:
        config = yaml.safe_load(yml)

    graph_dst = Path(config['destination']) / config['data_division'] / 'graph'
    graph_dst.mkdir(parents=True, exist_ok=True)
    graph_dst = graph_dst / 'confusion_matrix'
    graph_dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config['creature_data_destination'] + 'csv/class_sum.csv', encoding='shift-jis')
    fig, axes = plt.subplots(figsize=(22,23))
    cm = pd.DataFrame(data=accuracy_rate, index=df['class'], columns=df['class'])
    
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap = 'Blues', vmax=1, vmin=0)
    
    #graphの書式設定
    plt.xlabel("Pre", fontsize=15, rotation=0)
    plt.xticks(fontsize=15)
    plt.ylabel("Ans", fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(graph_dst / (config['heat_map_name'] + str(idx) + '.png')) 
    plt.close()
