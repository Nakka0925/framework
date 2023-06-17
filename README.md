# ミトコンドリアゲノムDNAを用いた、深層学習による生物種の分類


## 精度

|  accuracy  |  f1score  |
| ---- | ---- |
|  0.88  | 0.48  |

# 環境
- Ubunts 20.04 LTS
- Python 3.8.10
- Tensorflow 2.8.0


## 使い方

1. 初期設定

```console
$ python tools/initialize.py
```
`train_setting.yml`が生成されるので各自でパラメータを設定する

2. 画像データのダウンロード
```console
$ ./data_download.sh  
$ unzip img_data.zip  #展開
```

3. 学習の実行
```console
$ python train.py
```