use std::collections::HashSet;
use constants::*;

/// 連の大きさとダメの数を保持する構造体です。
/// 石がどの連に属するかについてはBoardのidが管理します。
/// 座標が入る場合、すべて、拡張碁盤の線形座標です。
pub struct StoneGroup {
    lib_cnt: usize,
    size: usize,
    v_atr: usize, // アタリの際の最後のダメの座標
    libs: HashSet<usize>,
}

impl StoneGroup {
    pub fn new() -> Self {
        StoneGroup {
            lib_cnt: VNULL,
            size: VNULL,
            v_atr: VNULL,
            libs: HashSet::new(),
        }
    }

    #[inline]
    pub fn get_size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn get_lib_cnt(&self) -> usize {
        self.lib_cnt
    }

    #[inline]
    pub fn get_v_atr(&self) -> usize {
        self.v_atr
    }

    pub fn clear(&mut self, stone: bool) {
        self.lib_cnt = if stone { 0 } else { VNULL };
        self.size = if stone { 1 } else { VNULL };
        self.v_atr = VNULL;
        self.libs.clear();
    }

    /// ダメを追加します。
    pub fn add(&mut self, v: usize) {
        if self.libs.contains(&v) {
            return;
        }
        self.libs.insert(v);
        self.lib_cnt += 1;
        self.v_atr = v;
    }

    /// ダメを削除します。
    pub fn sub(&mut self, v: usize) {
        if !self.libs.contains(&v) {
            return;
        }
        self.libs.remove(&v);
        self.lib_cnt -= 1;
    }

    /// 蓮otherをマージします。
    pub fn merge(&mut self, other: &Self) {
        use std::ops::BitOr;
        self.libs = self.libs.bitor(&other.libs);
        self.lib_cnt = self.libs.len();
        self.size += other.size;
        if self.lib_cnt == 1 {
            self.v_atr = *self.libs.iter().next().unwrap();
        }
    }

    pub fn copy_to(&self, dest: &mut Self) {
        dest.lib_cnt = self.lib_cnt;
        dest.size = self.size;
        dest.v_atr = self.v_atr;
        dest.libs = self.libs.clone();
    }
}
