# ミトコンドリアゲノムDNAを用いた、深層学習による生物種の分類

## 概要

ミトコンドリアゲノムDNAをグラフ画像化したものを入力とした、「綱」における生物種の分類

## 精度

|  accuracy  |  f1score  |
| ---- | ---- |
|  0.88  | 0.48  |

## 使い方

1. 初期設定

```console
$ python tools/initialize.py
```
`train_setting.yml`が生成されるので各自でパラメータを設定する

2. 学習の実行
```console
$ python train.py
```