use std::hash::Hash;
use std::collections::HashMap;

/// 配列を指定の値vで埋めます。
pub fn fill<T: Copy>(array: &mut [T], v: T) {
    for e in array.iter_mut() {
        *e = v;
    }
}

#[test]
fn test_fill() {
    let mut array = [0; 3];
    fill(&mut array, 1);
    assert!(array[0] == 1 && array[1] == 1 && array[2] == 1);
}

/// 配列の中で最も出現頻度の高い要素を返します。
/// 空の配列を渡すとpanicします。
pub fn most_common<T: Hash + Eq>(array: &[T]) -> &T {
    let mut map = HashMap::new();
    for e in array {
        let counter = map.entry(e).or_insert(0);
        *counter += 1;
    }
    let mut iter = map.into_iter();
    let first = iter.next().unwrap();
    iter.fold(first, |s, e| if s.1 >= e.1 { s } else { e }).0
}

#[test]
fn test_most_common() {
    assert_eq!(most_common(&[0, 1, 1]), &1);
}
