# 汎用人工知能HTMを用いた話者認識システム
　[HTM(Hierarchical Temporal Memory)](https://numenta.com/assets/pdf/whitepapers/hierarchical-temporal-memory-cortical-learning-algorithm-0.2.1-jp.pdf)は、人間の大脳新皮質のしくみを計算機上で再現するというコンセプトで開発されている人工知能です。
このリポジトリでは、HTMを使用した話者認識システムを実現するための試みを行っています。

# １．分類デモ
①親ディレクトリにReddotsデータセットを配置した状態で、以下を実行  
②classification reportとconfusion matrixが表示される
```
$ python recognition.py
```
  
# ２．分類スコアをもとにした最適パラメータ探索（粒子群最適化）
①以下を実行（例では7スレッド・RAM使用上限14GB・粒子数100で最適化）  
②recognition_ae/下に最適パラメータを記述したファイルがされる
```
$ cd experiment
$ python swarming.py -n 7 --memory_limit 14 -v --swarming 100 recognition.py
```