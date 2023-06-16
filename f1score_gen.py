import cv2
import numpy as np
from sklearn.metrics import f1_score
from graph_create import create_heatmap

def f1score_cacl(test_x, test_t, model, idx):
    """f1scoreを算出する

    Args:
        test_x (list) : テストデータのimgデータpathのlist
        test_t (list) : テストデータのlabelのlist
        model (model) : 学習モデル
        idx (int) : テストデータにするインデックス

    return:
        f1score (list) : モデルの予測と正解データでのf1scoreのlist
    """
    
    images = []

    for path in test_x:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(1 - np.asarray(img, dtype=np.float32) / 255)

    images = np.asarray(images, dtype=np.float32)
    label = np.asarray(test_t, dtype=np.float32)

    predictions = model.predict(images)
    predictions = np.argmax(predictions,axis=1)

    f1score = f1_score(label, predictions, average='macro')
    create_heatmap(label, predictions, idx)

    return f1score