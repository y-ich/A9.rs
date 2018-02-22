/// コミです。
pub const KOMI: f32 = 7.0;

/// 碁盤のサイズです。
pub const BSIZE: usize = 9;

/// 外枠を持つ拡張碁盤のサイズです。
pub const EBSIZE: usize = BSIZE + 2;

/// 碁盤の交点の数です。
pub const BVCNT: usize = BSIZE * BSIZE;

/// 拡張碁盤の交点の数です。
pub const EBVCNT: usize = EBSIZE * EBSIZE;

/// パスを表す線形座標です。通常の着手は拡張碁盤の線形座標で表します。
// TODO - 着手のために列挙型を作ったほうが関数のシグニチャは読みやすい。
pub const PASS: usize = EBVCNT;

/// 未使用の線形座標です。
// TODO - 該当する場所にOption<usize>を使ったほうが関数のシグニチャは読みやすい。
pub const VNULL: usize = EBVCNT + 1;
