use constants::*;

pub const X_LABELS: [char; 20] = [
    '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T',
];

/// 拡張碁盤の線形座標をxy座標に変換します。
#[inline]
pub fn ev2xy(ev: usize) -> (u8, u8) {
    ((ev % EBSIZE) as u8, (ev / EBSIZE) as u8)
}

/// 碁盤のxy座標を拡張碁盤の線形座標に変換します。
#[inline]
pub fn xy2ev(x: u8, y: u8) -> usize {
    y as usize * EBSIZE + x as usize
}

/// 碁盤の線形座標を拡張碁盤の線形座標に変換します。
#[inline]
pub fn rv2ev(rv: usize) -> usize {
    if rv == BVCNT {
        PASS
    } else {
        rv % BSIZE + 1 + (rv / BSIZE + 1) * EBSIZE
    }
}

/// 拡張碁盤の線形座標を碁盤の線形座標に変換します。
#[inline]
pub fn ev2rv(ev: usize) -> usize {
    if ev == PASS {
        BVCNT
    } else {
        ev % EBSIZE - 1 + (ev / EBSIZE - 1) * BSIZE
    }
}

/// 拡張碁盤の線形座標を碁盤の座標の文字表現に変換します。
pub fn ev2str(ev: usize) -> String {
    if ev >= PASS {
        "pass".to_string()
    } else {
        let (x, y) = ev2xy(ev);
        let mut s = y.to_string();
        s.insert(0, X_LABELS[x as usize]);
        s
    }
}

/// 碁盤の座標の文字表現を拡張碁盤の線形座標に変換します。
pub fn str2ev(v: &str) -> usize {
    let v_str = v.to_uppercase();
    if v_str == "PASS" || v_str == "RESIGN" {
        PASS
    } else {
        let mut chars = v_str.chars();
        let first = chars.next().unwrap();
        let x = X_LABELS.iter().position(|&e| e == first).unwrap() as u8;
        let y = chars.collect::<String>().parse::<u8>().unwrap();
        xy2ev(x, y)
    }
}
