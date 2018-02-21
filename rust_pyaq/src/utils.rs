pub fn most_common(array: &[f32]) -> f32 {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    for e in array {
        let counter = map.entry((e * 2.0) as i16).or_insert(0);
        *counter += 1;
    }
    map.into_iter().fold((0, -1), |s, e| if s.1 >= e.1 { s } else { e }).0 as f32 / 2.0
}
