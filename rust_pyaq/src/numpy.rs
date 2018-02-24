//! numpyとは全く関係ありませんが、Pyaqで使われているnumpyのヘルパー関数の代わりの関数を提供します

/// 配列の中の最大値のインデックスを返します。
/// 空の配列を渡すとpanicします。
pub fn argmax<T: PartialOrd>(array: &[T]) -> usize {
    let mut iter = array.iter().enumerate();
    let first = iter.next().unwrap();
    iter.fold(first, |m, v| if v.1 > m.1 { v } else { m }).0
}

#[test]
fn test_argmax() {
    assert_eq!(argmax(&[0, 1, 2, 1, 0]), 2);
}

/// 配列をソートした時の元のインデックスのVecを返します。
/// つまり[array[戻り値[0]], array[戻り値[1]], ...]が配列のソートになる関係です。
/// reverseがtrueの時は降順です。
pub fn argsort<T: PartialOrd>(array: &[T], reverse: bool) -> Vec<usize> {
    use std::cmp::Ordering;

    let mut en: Vec<(usize, &T)> = array.iter().enumerate().collect();
    en.sort_by(|a, b| {
        if reverse {
            b.1.partial_cmp(a.1).unwrap_or(Ordering::Equal)
        } else {
            a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)
        }
    });
    en.into_iter().map(|e| e.0).collect()
}

#[test]
fn test_argsort() {
    let a = argsort(&[3, 5, 1, 2], false);
    assert!(a[0] == 2 && a[1] == 3 && a[2] == 0 && a[3] == 1);
}

#[test]
fn test_argsort_rev() {
    let a = argsort(&[3, 5, 1, 2], true);
    assert!(a[0] == 1 && a[1] == 0 && a[2] == 3 && a[3] == 2);
}
