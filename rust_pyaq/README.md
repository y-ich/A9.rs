# rust_pyaq
rust_pyaqは、9路盤囲碁AI [Pyaq](https://github.com/ymgaq/Pyaq) のRustへの移植です。
対局部分のみで学習部分の移植はしていないです。

## コンパイルの仕方
TensorFlowのRust APIを使っています。このままでもコンパイルできるみたいですが、TensorFlow自体のコンパイルが始まるので重いこと、cargo cleanするたびにこれが起こるようなのでこのままコンパイルはお薦めしません。

別途、libtensorflow.soを用意してダイナミックリンクのパスに置いてください。

macOSでHomeBrewを利用されているかたはHomeBrewでインストールできます。

```
brew install libtensorflow
```

本体はnightlyでコンパイルする必要があります。(box syntax使用のため)
```
cargo +nightly build --release
```

## 遊び方
rust_pyaqの下にfrozen_model.pbを用意してください。
```
cargo +nightly run --release -- -h
```
でオプションの説明が出ます。

## ライセンス
MIT
