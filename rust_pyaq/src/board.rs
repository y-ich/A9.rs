use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use tensorflow as tf;
use constants::*;
use stone_group::StoneGroup;

const KEEP_PREV_CNT: usize = 2;
const FEATURE_CNT: usize = KEEP_PREV_CNT * 2 + 3;  // 7
const X_LABELS: [char; 19] = ['A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','Q','R','S','T'];


/// 拡張碁盤の線形座標をxy座標に変換します。
#[inline]
fn ev2xy(ev: usize) -> (usize, usize) {
    (ev % EBSIZE, ev / EBSIZE)
}

/// 碁盤のxy座標を拡張碁盤の線形座標に変換します。
#[inline]
fn xy2ev(x: usize, y: usize) -> usize {
    y * EBSIZE + x
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
fn ev2rv(ev: usize) -> usize {
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
        s.insert(0, X_LABELS[x - 1]);
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
        let x = X_LABELS.iter().position(|&e| e == first).unwrap() + 1;
        let y = chars.collect::<String>().parse::<usize>().unwrap();
        xy2ev(x, y)
    }
}

/// 拡張碁盤の線形座標vの点の隣接点の線形座標の配列を返します。
#[inline]
fn neighbors(v: usize) -> [usize; 4] {
    [v + 1, v + EBSIZE, v - 1, v - EBSIZE]
}

/// 拡張碁盤の線形座標vの点の斜め隣接点の線形座標の配列を返します。
fn diagonals(v: usize) -> [usize; 4] {
    [v + EBSIZE - 1, v + EBSIZE - 1, v - EBSIZE - 1, v - EBSIZE + 1]
}


/// 着手に関するエラーです。
pub enum Error {
    Illegal,
    FillEye,
}


/// 石の色や手番を表す列挙型です。
#[derive(Clone, Copy, PartialEq, Hash)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    fn opponent(&self) -> Self {
        match *self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}


// 交点の状態を表す列挙型です。
#[derive(Clone, Copy, PartialEq, Hash)]
enum Intersection {
    Stone(Color),
    Empty,
    Exterior, // 盤の外
}

impl Intersection {
    #[inline]
    fn to_usize(&self) -> usize {
        match *self {
            Intersection::Stone(c) => c as usize,
            Intersection::Empty => 2,
            Intersection::Exterior => 3,
        }
    }
}


/// 盤上の局面を表し、操作するための構造体です。
pub struct Board {
    state: [Intersection; EBVCNT], // 盤上の状態
    id: [usize; EBVCNT], // StoneGroupのid
    next: [usize; EBVCNT], // 同じStoneGroupの次の石の座標
    sg: Vec<StoneGroup>, // TODO - Copyでない構造体の配列の初期化の方法がわからなかったので、Vecにした
    prev_state: [[Intersection; EBVCNT]; KEEP_PREV_CNT],
    ko: usize,
    turn: Color,
    move_cnt: usize,
    prev_move: usize,
    remove_cnt: usize,
    history: Vec<usize>,
}

fn initialized_sg() -> Vec<StoneGroup> {
    let mut result = Vec::with_capacity(EBVCNT);
    for _ in 0..EBVCNT {
        result.push(StoneGroup::new());
    }
    result
}

impl Board {
    pub fn new() -> Self {
        let mut result = Self {
            state: [Intersection::Exterior; EBVCNT],
            id: [0; EBVCNT],
            next: [0; EBVCNT],
            sg: initialized_sg(),
            prev_state: [[Intersection::Exterior; EBVCNT]; KEEP_PREV_CNT],
            ko: VNULL,
            turn: Color::Black,
            move_cnt: 0,
            prev_move: VNULL,
            remove_cnt: 0,
            history: Vec::with_capacity(BVCNT * 2),
        };
        result.clear();
        result
    }

    #[inline]
    pub fn get_move_cnt(&self) -> usize {
        self.move_cnt
    }

    #[inline]
    pub fn get_prev_move(&self) -> usize {
        self.prev_move
    }

    #[inline]
    pub fn get_history(&self) -> &Vec<usize> {
        &self.history
    }

    pub fn clear(&mut self) {
        for x in 1..BSIZE + 1 {
            for y in 1..BSIZE + 1 {
                self.state[xy2ev(x, y)] = Intersection::Empty;
            }
        }
        for (i, e) in self.id.iter_mut().enumerate() {
            *e = i;
        }
        for (i, e) in self.next.iter_mut().enumerate() {
            *e = i;
        }
        for e in self.sg.iter_mut() {
            e.clear(false);
        }
        for e in self.prev_state.iter_mut() {
            *e = self.state;
        }
        self.ko = VNULL;
        self.turn = Color::Black;
        self.move_cnt = 0;
        self.prev_move = VNULL;
        self.remove_cnt = 0;
        self.history.clear();
    }

    pub fn copy_to(&self, dest: &mut Self) {
        dest.state = self.state;
        dest.id = self.id;
        dest.next = self.next;
        for (i, e) in dest.sg.iter_mut().enumerate() {
            self.sg[i].copy_to(e);
        }
        dest.prev_state = self.prev_state;
        dest.ko = self.ko;
        dest.turn = self.turn;
        dest.move_cnt = self.move_cnt;
        dest.remove_cnt = self.remove_cnt;
        dest.history = self.history.clone();
    }

    /// 線形座標vの位置の石とその連を上げます。
    pub fn remove(&mut self, v: usize) {
        let mut v_tmp = v;
        loop {
            self.remove_cnt += 1;
            self.state[v_tmp] = Intersection::Empty;
            self.id[v_tmp] = v_tmp;
            for &nv in &neighbors(v_tmp) {
                self.sg[self.id[nv]].add(v_tmp);
            }
            let v_next = self.next[v_tmp];
            self.next[v_tmp] = v_tmp;
            v_tmp = v_next;
            if v_tmp == v {
                break;
            }
        }
    }

    /// v1が属する連とv2がぞくする連をマージします。
    pub fn merge(&mut self, v1: usize, v2: usize) {
        let mut id_base = self.id[v1];
        let mut id_add = self.id[v2];
        if self.sg[id_base].get_size() < self.sg[id_add].get_size() {
            let tmp = id_base;
            id_base = id_add;
            id_add = tmp;
        }

        // 以下は、self.sg[id_base].merge(&self.sg[id_add]);の意味
        if id_base < id_add {
            let (front, back) = self.sg.split_at_mut(id_add);
            front[id_base].merge(&back[0]);
        } else {
            let (front, back) = self.sg.split_at_mut(id_base);
            back[0].merge(&front[id_add]);
        }

        let mut v_tmp = id_add;
        loop {
            self.id[v_tmp] = id_base;
            v_tmp = self.next[v_tmp];
            if v_tmp == id_add {
                break;
            }
        }
        let tmp = self.next[v1];
        self.next[v1] = self.next[v2];
        self.next[v2] = tmp;
    }

    /// 着手します。
    pub fn place_stone(&mut self, v: usize) {
        let stone_color = Intersection::Stone(self.turn);
        self.state[v] = stone_color;
        self.id[v] = v;
        self.sg[self.id[v]].clear(true);
        for &nv in &neighbors(v) {
            if self.state[nv] == Intersection::Empty {
                self.sg[self.id[v]].add(nv);
            } else {
                self.sg[self.id[nv]].sub(v);
            }
        }
        for &nv in &neighbors(v) {
            if self.state[nv] == stone_color && self.id[nv] != self.id[v] {
                self.merge(v, nv);
            }
        }
        self.remove_cnt = 0;
        let opponent_stone = Intersection::Stone(self.turn.opponent());
        for &nv in &neighbors(v) {
            if self.state[nv] == opponent_stone && self.sg[self.id[nv]].get_lib_cnt() == 0 {
                self.remove(nv);
            }
        }
    }

    /// 着手禁止点でないか調べます。
    pub fn legal(&self, v: usize) -> bool {
        if v == PASS {
            return true;
        } else if v == self.ko || self.state[v] != Intersection::Empty {
            return false;
        }

        let mut stone_cnt = [0, 0];
        let mut atr_cnt = [0, 0];
        for &nv in &neighbors(v) {
            let c = self.state[nv];
            match c {
                Intersection::Empty => {
                    return true; // 一つでもダメが空いていれば着手可能
                },
                Intersection::Stone(c) => {
                    stone_cnt[c as usize] += 1;
                    if self.sg[self.id[nv]].get_lib_cnt() == 1 {
                        atr_cnt[c as usize] += 1;
                    }
                },
                _ => {},
            }
        }
        atr_cnt[self.turn.opponent() as usize] != 0 || // 相手の石が取れるか
        atr_cnt[self.turn as usize] < stone_cnt[self.turn as usize] // アタリでない石と繋がるか
    }

    /// plの眼形か調べます。
    /// ポン抜きの形で、アタリでない相手の石で欠け目にされている時以外を眼形と定義します。
    // TODO - この条件でいいの？
    pub fn eyeshape(&self, v: usize, pl: Color) -> bool {
        if v == PASS {
            return false;
        }
        for &nv in &neighbors(v) {
            let c = self.state[nv];
            if c == Intersection::Empty || c == Intersection::Stone(pl.opponent()) {
                return false; // 周りに一つでも空点や敵の石があれば眼形でない
            }
        }
        let mut diag_cnt = [0, 0, 0, 0];
        for &nv in &diagonals(v) {
            diag_cnt[self.state[nv].to_usize()] += 1;
        }
        let wedge_cnt = diag_cnt[pl.opponent() as usize] + if diag_cnt[3] > 0 { 1 } else { 0 };
        if wedge_cnt == 2 {
            for &nv in &diagonals(v) {
                if self.state[nv] == Intersection::Stone(pl.opponent()) &&
                    self.sg[self.id[nv]].get_lib_cnt() == 1 &&
                    self.sg[self.id[nv]].get_v_atr() != self.ko {
                    return true;
                }
            }
        }
        return wedge_cnt < 2;
    }

    /// 着手可能か調べて、可能ならば着手して必要な内部状態を更新します。
    pub fn play(&mut self, v: usize, not_fill_eye: bool) -> Result<(), Error> {
        if !self.legal(v) {
            return Err(Error::Illegal);
        }
        if not_fill_eye && self.eyeshape(v, self.turn) {
            return Err(Error::FillEye);
        }
        for i in (0..KEEP_PREV_CNT - 1).rev() {
            self.prev_state[i + 1] = self.prev_state[i];
        }
        self.prev_state[0] = self.state;
        if v == PASS {
            self.ko = VNULL;
        } else {
            self.place_stone(v);
            let id = self.id[v];
            self.ko = VNULL;
            if self.remove_cnt == 1 &&
                self.sg[id].get_lib_cnt() == 1 &&
                self.sg[id].get_size() == 1 {
                self.ko = self.sg[id].get_v_atr();
            }
        }
        self.prev_move = v;
        self.history.push(v);
        self.turn = self.turn.opponent();
        self.move_cnt += 1;
        Ok(())
    }

    /// 眼を埋めないようにランダムプレイします。
    pub fn random_play(&mut self) -> usize {
        use rand::{thread_rng, Rng};

        let mut empty_list: Vec<usize> = self.state.iter().enumerate()
            .filter_map(|(i, &e)| if e == Intersection::Empty { Some(i) } else { None })
            .collect();
        let mut rng = thread_rng();
        rng.shuffle(&mut empty_list);
        for &v in &empty_list {
            if let Ok(_) = self.play(v, true) {
                return v;
            }
        }
        let _ = self.play(PASS, true);
        PASS
    }

    /// 現局面のスコアを返します。
    /// 盤上の石の数と一方の石のみに隣接する空点の数の差がスコアです。
    /// なので、死に石すべてを上げて十分に陣地を埋めてから使います。
    /// 死に石すべてを上げて十分に陣地を埋めるにはメソッドrolloutを使います。
    // TODO - Tromp-Taylorルールに変えるべきかどうかの検討
    pub fn score(&self) -> f32 {
        let mut stone_cnt = [0, 0];
        for v in (0..BVCNT).map(rv2ev) {
            let s = self.state[v];
            if let Intersection::Stone(c) = s {
                stone_cnt[c as usize] += 1;
            } else {
                let mut nbr_cnt = [0, 0, 0, 0];
                for &nv in &neighbors(v) {
                    nbr_cnt[self.state[nv].to_usize()] += 1;
                }
                if nbr_cnt[Color::White as usize] > 0 && nbr_cnt[Color::Black as usize] == 0 {
                    stone_cnt[Color::White as usize] += 1;
                } else if nbr_cnt[Color::Black as usize] > 0 && nbr_cnt[Color::White as usize] == 0 {
                    stone_cnt[Color::Black as usize] += 1;
                }
            }
        }
        (stone_cnt[1] - stone_cnt[0]) as f32 - KOMI
    }

    /// 原始モンテカルロでロールアウトします。
    /// 死に石すべてを上げて十分に陣地を埋めるのに使います。
    pub fn rollout(&mut self, show_board: bool) {
        while self.move_cnt < EBVCNT * 2 {
            let prev_move = self.prev_move;
            let mov = self.random_play();
            if show_board && mov != PASS {
                eprintln!("\nmove count={}", self.move_cnt);
                self.showboard();
            }
            if prev_move == PASS && mov == PASS {
                break;
            }
        }
    }

    pub fn showboard(&self) {
        fn print_xlabel() {
            let mut line_str = "  ".to_string();
            for x in 0..BSIZE {
                line_str.push_str(&format!(" {} ", X_LABELS[x]));
            }
            eprintln!("{}", line_str);
        }
        print_xlabel();
        for y in (1..BSIZE + 1).rev() {
            let mut line_str = format!("{:>2}", y);
            for x in 1..BSIZE + 1 {
                let v = xy2ev(x, y);
                let x_str = match self.state[v] {
                    Intersection::Stone(c) => {
                        let stone_str = match c {
                            Color::White => "O",
                            Color::Black => "X",
                        };
                        if v == self.prev_move {
                            format!("[{}]", stone_str)
                        } else {
                            format!(" {} ", stone_str)
                        }
                    },
                    Intersection::Empty => {
                        " . ".to_string()
                    },
                    _ => {
                        " ? ".to_string()
                    }
                };
                line_str.push_str(&x_str);
            }
            line_str.push_str(&format!("{:>2}", y));
            eprintln!("{}", line_str);
        }
        print_xlabel();
        eprintln!("");
    }

    /// ニューラルネットワークへの入力を返します。
    pub fn feature(&self) -> tf::Tensor<f32> {
        #[inline]
        fn index(p: usize, f: usize) -> usize {
            p * FEATURE_CNT + f
        }

        let mut feature_ = tf::Tensor::new(&[BVCNT as u64, FEATURE_CNT as u64]);
        let my = Intersection::Stone(self.turn);
        let opp = Intersection::Stone(self.turn.opponent());
        for p in 0..BVCNT {
            *feature_.get_mut(index(p, 0)).unwrap() =
                if self.state[rv2ev(p)] == my { 1.0 } else { 0.0 };
        }
        for p in 0..BVCNT {
            *feature_.get_mut(index(p, 1)).unwrap() =
                if self.state[rv2ev(p)] == opp { 1.0 } else { 0.0 };
        }
        for i in 0..KEEP_PREV_CNT {
            for p in 0..BVCNT {
                *feature_.get_mut(index(p, (i + 1) * 2)).unwrap() =
                    if self.prev_state[i][rv2ev(p)] == my { 1.0 } else { 0.0 };
            }
            for p in 0..BVCNT {
                *feature_.get_mut(index(p, (i + 1) * 2 + 1)).unwrap() =
                    if self.prev_state[i][rv2ev(p)] == opp { 1.0 } else { 0.0 };
            }
        }
        for p in 0..BVCNT {
            *feature_.get_mut(index(p, FEATURE_CNT - 1)).unwrap() = my.to_usize() as f32;
        }

        feature_
    }

    /// 局面のハッシュを返します。
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.state.hash(&mut hasher);
        let h1 = hasher.finish();
        self.prev_state[0].hash(&mut hasher);
        let h2 = hasher.finish();

        (h1 ^ h2) ^ self.turn as u64
    }

    /// 局面の情報を返します。
    pub fn info(&self) -> (u64, usize, Vec<usize>) {
        let mut cand_list: Vec<usize> = self.state.iter().enumerate()
            .filter_map(|(v, &e)|
                if e == Intersection::Empty && self.legal(v) && !self.eyeshape(v, self.turn) {
                    Some(ev2rv(v))
                } else {
                    None
                }
            ).collect();
        cand_list.push(ev2rv(PASS));
        (self.hash(), self.move_cnt, cand_list)
    }
}
