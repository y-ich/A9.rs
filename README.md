# A9.js
A9.jsは、山口祐さんが開発された9路盤囲碁AI [Pyaq](https://github.com/ymgaq/Pyaq) をブラウザ上で動かそうというプロジェクトです。

## 予定と進捗
以下の手順で開発を進める予定です。
1. Pyaqの対局部分をRustに移植する(rust_pyaq)
1. WASMにコンパイルしてブラウザで動かす
1. Pyaqのニューラルネットワーク部分をWebDNN上で動かす
1. RustコードからWebDNNを呼び出して高速化する

現在、1がざっくりと完了しました。

## ライセンス
MIT
