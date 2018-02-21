use std::cmp::Ordering;

pub fn argmax<T: PartialOrd>(array: &[T]) -> Option<usize> {
    let mut iter = array.iter().enumerate();
    if let Some(first) = iter.next() {
        Some(iter.fold(first, |m, v| if v.1 > m.1 { v } else { m }).0)
    } else {
        None
    }
}

pub fn argsort<T: PartialOrd>(array: &[T], reverse: bool) -> Vec<usize> {
    let mut en: Vec<(usize, &T)> = array.iter().enumerate().collect();
    en.sort_by(|a, b| if reverse {
        b.1.partial_cmp(a.1).unwrap_or(Ordering::Equal)
    } else {
        a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)
    });
    en.into_iter().map(|e| e.0).collect()
}
