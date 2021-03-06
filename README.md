# A9.js
A9.jsは、山口祐さんが開発された9路盤囲碁AI [Pyaq](https://github.com/ymgaq/Pyaq) をブラウザ上で動かそうというプロジェクトです。

(Pyaqは、AQ(2.1.1)とは異なり、高速ロールアウトを排除したAlphaGo Zeroタイプのプログラムだと思います。
9路盤AlphaGo ZeroのPython実装と言えるのではないでしょうか。
ただ、公開されている学習済みネットワークは自己対戦ではなく棋譜からの学習のようです。)

## 予定と進捗
以下の手順で開発を進める予定です。
1. Pyaqの対局部分をRustに移植する(rust_pyaq)
1. ~~WASMにコンパイルしてブラウザで動かす~~ libtensorflowの部分があるのでこれは無理でした
1. Pyaqのニューラルネットワーク部分をWebDNN上で動かす
1. RustコードからWebDNNを呼び出して高速化する

現在、3が終了して、4で問題を抱えています。

## 謝辞
移植しやすいコードを公開頂いた山口祐さんに感謝致します。

## ライセンス
MIT
