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

本体はnightlyでコンパイルする必要があります。
```
cargo +nightly build --release
```

## 性能
CPU版で4.5倍ぐらいな雰囲気でした。(iMac Late 2012, 2.9GHz Core i5。初手)

## ライセンス
MIT
