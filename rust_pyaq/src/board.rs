use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::ops::BitOr;
use tensorflow as tf;

pub const BSIZE: usize = 9;
pub const EBSIZE: usize = BSIZE + 2;
pub const BVCNT: usize = BSIZE * BSIZE;
pub const EBVCNT: usize = EBSIZE * EBSIZE;
pub const PASS: usize = EBVCNT;
pub const VNULL: usize = EBVCNT + 1;
pub const KOMI: f32 = 7.0;
const KEEP_PREV_CNT: usize = 2;
pub const FEATURE_CNT: usize = KEEP_PREV_CNT * 2 + 3;  // 7
const X_LABELS: [char; 19] = ['A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','Q','R','S','T'];


fn ev2xy(ev: usize) -> (usize, usize) {
    (ev % EBSIZE, ev / EBSIZE)
}

fn xy2ev(x: usize, y: usize) -> usize {
    y * EBSIZE + x
}

pub fn rv2ev(rv: usize) -> usize {
    if rv == BVCNT {
        PASS
    } else {
        rv % BSIZE + 1 + (rv / BSIZE + 1) * EBSIZE
    }
}

pub fn ev2rv(ev: usize) -> usize {
    if ev == PASS {
        BVCNT
    } else {
        ev % EBSIZE - 1 + (ev / EBSIZE - 1) * BSIZE
    }
}

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

fn neighbors(v: usize) -> [usize; 4] {
    [v + 1, v + EBSIZE, v - 1, v - EBSIZE]
}

fn diagonals(v: usize) -> [usize; 4] {
    [v + EBSIZE - 1, v + EBSIZE - 1, v - EBSIZE - 1, v - EBSIZE + 1]
}


pub enum Error {
    Illegal,
    FillEye,
}

struct StoneGroup {
    lib_cnt: usize,
    size: usize,
    v_atr: usize,
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

    pub fn clear(&mut self, stone: bool) {
        self.lib_cnt = if stone { 0 } else { VNULL };
        self.size = if stone { 1 } else { VNULL };
        self.v_atr = VNULL;
        self.libs.clear();
    }

    pub fn add(&mut self, v: usize) {
        if self.libs.contains(&v) {
            return;
        }
        self.libs.insert(v);
        self.lib_cnt += 1;
        self.v_atr = v;
    }

    pub fn sub(&mut self, v: usize) {
        if !self.libs.contains(&v) {
            return;
        }
        self.libs.remove(&v);
        self.lib_cnt -= 1;
    }

    pub fn merge(&mut self, other: &Self) {
        self.libs = self.libs.bitor(&other.libs);
        self.lib_cnt = self.libs.len();
        self.size += other.size;
        if self.lib_cnt == 1 {
            for lib in self.libs.iter() {
                self.v_atr = *lib;
            }
        }
    }

    pub fn copy_to(&self, dest: &mut Self) {
        dest.lib_cnt = self.lib_cnt;
        dest.size = self.size;
        dest.v_atr = self.v_atr;
        dest.libs = self.libs.clone();
    }
}


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


#[derive(Clone, Copy, PartialEq, Hash)]
enum Intersection {
    Stone(Color),
    Empty,
    Exterior,
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


pub struct Board {
    color: [Intersection; EBVCNT],
    id: [usize; EBVCNT],
    next: [usize; EBVCNT],
    sg: Vec<StoneGroup>,
    prev_color: [[Intersection; EBVCNT]; KEEP_PREV_CNT],
    ko: usize,
    turn: Color,
    pub move_cnt: usize,
    pub prev_move: usize,
    remove_cnt: usize,
    pub history: Vec<usize>,
}

impl Board {
    fn inialized_sg() -> Vec<StoneGroup> {
        let mut result = Vec::with_capacity(EBVCNT);
        for _ in 0..EBVCNT {
            result.push(StoneGroup::new());
        }
        result
    }

    pub fn new() -> Self {
        let mut result = Self {
            color: [Intersection::Exterior; EBVCNT],
            id: [0; EBVCNT],
            next: [0; EBVCNT],
            sg: Self::inialized_sg(),
            prev_color: [[Intersection::Exterior; EBVCNT]; KEEP_PREV_CNT],
            ko: VNULL,
            turn: Color::Black,
            move_cnt: 0,
            prev_move: VNULL,
            remove_cnt: 0,
            history: Vec::new(),
        };
        result.clear();
        result
    }

    pub fn clear(&mut self) {
        for x in 1..BSIZE + 1 {
            for y in 1..BSIZE + 1 {
                self.color[xy2ev(x, y)] = Intersection::Empty;
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
        for e in self.prev_color.iter_mut() {
            *e = self.color;
        }
        self.ko = VNULL;
        self.turn = Color::Black;
        self.move_cnt = 0;
        self.prev_move = VNULL;
        self.remove_cnt = 0;
        self.history.clear();
    }

    pub fn copy_to(&self, dest: &mut Self) {
        dest.color = self.color;
        dest.id = self.id;
        dest.next = self.next;
        for (i, e) in dest.sg.iter_mut().enumerate() {
            self.sg[i].copy_to(e);
        }
        dest.prev_color = self.prev_color;
        dest.ko = self.ko;
        dest.turn = self.turn;
        dest.move_cnt = self.move_cnt;
        dest.remove_cnt = self.remove_cnt;
        dest.history = self.history.clone();
    }

    pub fn remove(&mut self, v: usize) {
        let mut v_tmp = v;
        loop {
            self.remove_cnt += 1;
            self.color[v_tmp] = Intersection::Empty;
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

    pub fn merge(&mut self, v1: usize, v2: usize) {
        let mut id_base = self.id[v1];
        let mut id_add = self.id[v2];
        if self.sg[id_base].size < self.sg[id_add].size {
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

    pub fn place_stone(&mut self, v: usize) {
        let stone_color = Intersection::Stone(self.turn);
        self.color[v] = stone_color;
        self.id[v] = v;
        self.sg[self.id[v]].clear(true);
        for &nv in &neighbors(v) {
            if self.color[nv] == Intersection::Empty {
                self.sg[self.id[v]].add(nv);
            } else {
                self.sg[self.id[nv]].sub(v);
            }
        }
        for &nv in &neighbors(v) {
            if self.color[nv] == stone_color && self.id[nv] != self.id[v] {
                self.merge(v, nv);
            }
        }
        self.remove_cnt = 0;
        let opponent_stone = Intersection::Stone(self.turn.opponent());
        for &nv in &neighbors(v) {
            if self.color[nv] == opponent_stone && self.sg[self.id[nv]].lib_cnt == 0 {
                self.remove(nv);
            }
        }
    }

    pub fn legal(&self, v: usize) -> bool {
        if v == PASS {
            return true;
        } else if v == self.ko || self.color[v] != Intersection::Empty {
            return false;
        }

        let mut stone_cnt = [0, 0];
        let mut atr_cnt = [0, 0];
        for &nv in &neighbors(v) {
            let c = self.color[nv];
            match c {
                Intersection::Empty => {
                    return true;
                },
                Intersection::Stone(c) => {
                    stone_cnt[c as usize] += 1;
                    if self.sg[self.id[nv]].lib_cnt == 1 {
                        atr_cnt[c as usize] += 1;
                    }
                },
                _ => {},
            }
        }
        atr_cnt[self.turn.opponent() as usize] != 0 || atr_cnt[self.turn as usize] < stone_cnt[self.turn as usize]
    }

    pub fn eyeshape(&self, v: usize, pl: Color) -> bool {
        if v == PASS {
            return false;
        }
        for &nv in &neighbors(v) {
            let c = self.color[nv];
            if c == Intersection::Empty || c == Intersection::Stone(pl.opponent()) {
                return false;
            }
        }
        let mut diag_cnt = [0, 0, 0, 0];
        for &nv in &diagonals(v) {
            diag_cnt[self.color[nv].to_usize()] += 1;
        }
        let wedge_cnt = diag_cnt[pl.opponent() as usize] + if diag_cnt[3] > 0 { 1 } else { 0 };
        if wedge_cnt == 2 {
            for &nv in &diagonals(v) {
                if self.color[nv] == Intersection::Stone(pl.opponent()) &&
                    self.sg[self.id[nv]].lib_cnt == 1 &&
                    self.sg[self.id[nv]].v_atr != self.ko {
                    return true;
                }
            }
        }
        return wedge_cnt < 2;
    }

    pub fn play(&mut self, v: usize, not_fill_eye: bool) -> Result<(), Error> {
        if !self.legal(v) {
            return Err(Error::Illegal);
        }
        if not_fill_eye && self.eyeshape(v, self.turn) {
            return Err(Error::FillEye);
        }
        for i in (0..KEEP_PREV_CNT - 1).rev() {
            self.prev_color[i + 1] = self.prev_color[i];
        }
        self.prev_color[0] = self.color;
        if v == PASS {
            self.ko = VNULL;
        } else {
            self.place_stone(v);
            let id = self.id[v];
            self.ko = VNULL;
            if self.remove_cnt == 1 &&
                self.sg[id].lib_cnt == 1 &&
                self.sg[id].size == 1 {
                self.ko = self.sg[id].v_atr;
            }
        }
        self.prev_move = v;
        self.history.push(v);
        self.turn = self.turn.opponent();
        self.move_cnt += 1;
        Ok(())
    }

    pub fn random_play(&mut self) -> usize {
        use rand::{thread_rng, Rng};

        let mut empty_list: Vec<usize> = self.color.iter()
            .enumerate()
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

    pub fn score(&self) -> f32 {
        let mut stone_cnt = [0, 0];
        for v in (0..BVCNT).map(rv2ev) {
            let s = self.color[v];
            if let Intersection::Stone(c) = s {
                stone_cnt[c as usize] += 1;
            } else {
                let mut nbr_cnt = [0, 0, 0, 0];
                for &nv in &neighbors(v) {
                    nbr_cnt[self.color[nv].to_usize()] + 1;
                }
                if nbr_cnt[0] > 0 && nbr_cnt[1] == 0 {
                    stone_cnt[0] += 1;
                } else if nbr_cnt[1] > 0 && nbr_cnt[0] == 0 {
                    stone_cnt[1] += 1;
                }
            }
        }
        (stone_cnt[1] - stone_cnt[0]) as f32 - KOMI
    }

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
                let x_str = match self.color[v] {
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

    pub fn feature(&self) -> tf::Tensor<f32> {
        #[inline]
        fn index(p: usize, f: usize) -> usize {
            p * FEATURE_CNT + f
        }

        let mut feature_ = tf::Tensor::new(&[BVCNT as u64, FEATURE_CNT as u64]);
        let my = Intersection::Stone(self.turn);
        let opp = Intersection::Stone(self.turn.opponent());
        for p in 0..BVCNT {
            *feature_.get_mut(index(p, 0)).unwrap() = if self.color[rv2ev(p)] == my { 1.0 } else { 0.0 };
        }
        for p in 0..BVCNT {
            *feature_.get_mut(index(p, 1)).unwrap() = if self.color[rv2ev(p)] == opp { 1.0 } else { 0.0 };
        }
        for i in 0..KEEP_PREV_CNT {
            for p in 0..BVCNT {
                *feature_.get_mut(index(p, (i + 1) * 2)).unwrap() = if self.prev_color[i][rv2ev(p)] == my { 1.0 } else { 0.0 };
            }
            for p in 0..BVCNT {
                *feature_.get_mut(index(p, (i + 1) * 2 + 1)).unwrap() = if self.prev_color[i][rv2ev(p)] == opp { 1.0 } else { 0.0 };
            }
        }
        for p in 0..BVCNT {
            *feature_.get_mut(index(p, FEATURE_CNT - 1)).unwrap() = my.to_usize() as f32;
        }

        feature_
    }

    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.color.hash(&mut hasher);
        let h1 = hasher.finish();
        self.prev_color[0].hash(&mut hasher);
        let h2 = hasher.finish();

        (h1 ^ h2) ^ self.turn as u64
    }

    pub fn info(&self) -> (u64, usize, Vec<usize>) {
        let mut cand_list = Vec::new();
        for (v, _) in self.color.iter().enumerate().filter(|&(_, &e)| e == Intersection::Empty) {
            if self.legal(v) && !self.eyeshape(v, self.turn) {
                cand_list.push(ev2rv(v));
            }
        }
        cand_list.push(ev2rv(PASS));
        (self.hash(), self.move_cnt, cand_list)
    }
}
