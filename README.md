# MoA-models

# 前処理
## Rank Gauss
・数値変数を正規分布に変換(ニューラルネットワークを使用する際の特徴量変換の手法として、 正規化や標準化よりも優れた性能を発揮すると言われています)

・外れ値の影響を軽減

・対象となる値を順位付けし、その順位を -1 ~ 1 の範囲に変換

・変換後の結果を分位関数に適用し、正規分布を取得

## PCA
・次元削減と特徴量の追加

## Variance Threshold
・分散が閾値以下のデータを特徴量から除外

# モデル
①MLP (with K Folds by Drug ID)
後半から開示されたDrug IDを使用。
また、nonscoredを含めたターゲットを用いて事前に学習した重みを用いて、学習を実施。
使用したモデルの中でもっとも良いスコア(public)を出力。

②MLP (with DenoisingAutoEncoder)
DAE(DenoisingAutoEncoder)を用いて特徴抽出を実施。また、バリデーションの損失はepoch=30程度から改善せず。
エポック数を増やすほどCVが改善したのでエポック数を大きめに設定(epoch=70)した（おそらく過学習）

③TabNet
tabnetは表計算用のディープラーニングモデル。コンペでは、事前学習のないMLPに劣っていたが、discussionでは予測結果の分布が違うことからアンサンブルに有用だと議論されていた。

④TabNet (with KMeans clustering)
前述の前処理に加えて、kmeansによる特徴抽出を実施。

⑤ResNet
resnetのアーキテクチャを模擬したモデル。

# 各モデルの相関ヒートマップ
![image](https://user-images.githubusercontent.com/90243284/163803564-4b59c26b-3c63-4846-9e70-205208011f80.png)

# スコア
![image](https://user-images.githubusercontent.com/90243284/163803640-2652e048-faec-439d-bbed-246a4812ead9.png)

