use std::collections::HashMap;
use std::time;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use tensorflow as tf;
use numpy as np;
use constants::*;
use board::*;
#[cfg(not(target_arch = "wasm32"))]
use neural_network::NeuralNetwork;

const MAX_NODE_CNT: usize = 16384; // 2 ^ 14
const EXPAND_CNT: usize = 8;

fn duration2float(d: time::Duration) -> f32 {
    d.as_secs() as f32 + d.subsec_nanos() as f32 / 1000_000_000.0
}

/// MCTSの各ノードです。
#[derive(Clone, Copy)] // 配列の初期化で楽するためにCopyにした
struct Node {
    // 各配列のBVCNT番目の要素はPASSに対応する着手
    mov: [usize; BVCNT + 1],
    prob: [f32; BVCNT + 1],
    value: [f32; BVCNT + 1],
    value_win: [f32; BVCNT + 1],
    visit_cnt: [usize; BVCNT + 1],
    next_id: [usize; BVCNT + 1],
    next_hash: [u64; BVCNT + 1],
    evaluated: [bool; BVCNT + 1],
    branch_cnt: usize,
    total_value: f32,
    total_cnt: usize,
    hash: u64,
    move_cnt: usize, // TODO - Option<usize>のほうがいいか
}

impl Node {
    pub fn new() -> Self {
        use std::mem::uninitialized;
        let mut node: Self = unsafe { uninitialized() };
        node.init_branch();
        node.clear();
        node
    }

    pub fn init_branch(&mut self) {
        use utils::fill;

        fill(&mut self.mov, VNULL);
        fill(&mut self.prob, 0.0);
        fill(&mut self.value, 0.0);
        fill(&mut self.value_win, 0.0);
        fill(&mut self.visit_cnt, 0);
        fill(&mut self.next_id, usize::max_value());
        fill(&mut self.next_hash, 0);
        fill(&mut self.evaluated, false);
    }

    pub fn clear(&mut self) {
        self.branch_cnt = 0;
        self.total_value = 0.0;
        self.total_cnt = 0;
        self.hash = 0;
        self.move_cnt = usize::max_value();
    }
}

/// ポンダーを実装するためのフラグ。未使用。
lazy_static! {
    static ref TREE_STOP: AtomicBool = AtomicBool::new(false);
}
/// UCB1のCp関連の係数？
static mut TREE_CP: f32 = 2.0;

/// MCTSを実行するワーカー構造体です。
pub struct Tree {
    main_time: f32,
    byoyomi: f32,
    left_time: f32,
    node: Box<[Node; MAX_NODE_CNT]>,
    node_cnt: usize,
    root_id: usize,
    root_move_cnt: usize,
    node_hashs: HashMap<u64, usize>,
    eval_cnt: usize,
    #[cfg(not(target_arch = "wasm32"))] nn: NeuralNetwork,
}

impl Tree {
    pub fn new(_ckpt_path: &str) -> Self {
        Self {
            main_time: 0.0,
            byoyomi: 1.0,
            left_time: 0.0,
            node: box [Node::new(); MAX_NODE_CNT],
            node_cnt: 0,
            root_id: 0,
            root_move_cnt: 0,
            node_hashs: HashMap::new(),
            eval_cnt: 0,
            #[cfg(not(target_arch = "wasm32"))]
            nn: NeuralNetwork::new(_ckpt_path),
        }
    }

    pub fn set_time(&mut self, main_time: f32, byoyomi: f32) {
        self.main_time = main_time;
        self.left_time = main_time;
        self.byoyomi = byoyomi;
    }

    pub fn set_left_time(&mut self, left_time: f32) {
        self.left_time = left_time;
    }

    pub fn clear(&mut self) {
        self.left_time = self.main_time;
        for nd in self.node.iter_mut() {
            nd.clear();
        }
        self.node_cnt = 0;
        self.root_id = 0;
        self.root_move_cnt = 0;
        self.node_hashs.clear();
        self.eval_cnt = 0;
        TREE_STOP.store(false, Ordering::Relaxed);
    }

    /// ニューラルネットワークを評価します。
    #[cfg(target_arch = "wasm32")]
    pub fn evaluate(&mut self, b: &Board) -> (Vec<f32>, Vec<f32>) {
        use stdweb::{Reference, UnsafeTypedArray};
        use stdweb::web::TypedArray;
        use stdweb::unstable::TryInto;
        let feature = &b.feature();
        let feature = unsafe { UnsafeTypedArray::new(feature) };
        let array: Vec<Reference> = js! { evaluate(@{feature}) }.try_into().unwrap();
        let mut iter = array
            .into_iter()
            .map(|e| e.downcast::<TypedArray<f32>>().unwrap().to_vec());
        (iter.next().unwrap(), iter.next().unwrap())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn evaluate(&mut self, b: &Board) -> (tf::Tensor<f32>, tf::Tensor<f32>) {
        self.nn.evaluate(&b.feature()).unwrap()
    }

    /// 不要なノード(現局面より手数が少ないノード)を削除します。
    fn delete_node(&mut self) {
        if self.node_cnt < MAX_NODE_CNT / 2 {
            return;
        }
        for i in 0..MAX_NODE_CNT {
            let mc = self.node[i].move_cnt;
            if mc < usize::max_value() && mc < self.root_move_cnt {
                self.node_hashs.remove(&self.node[i].hash);
                self.node[i].clear()
            }
        }
    }

    fn create_node(&mut self, b_info: (u64, usize, Vec<usize>), prob: &[f32]) -> usize {
        let hs = b_info.0;

        if self.node_hashs.contains_key(&hs) && self.node[self.node_hashs[&hs]].hash == hs
            && self.node[self.node_hashs[&hs]].move_cnt == b_info.1
        {
            return self.node_hashs[&hs];
        }

        let mut node_id = hs as usize % MAX_NODE_CNT;

        while self.node[node_id].move_cnt != usize::max_value() {
            node_id = if node_id + 1 < MAX_NODE_CNT {
                node_id + 1
            } else {
                0
            };
        }

        if let Some(hash) = self.node_hashs.get_mut(&hs) {
            *hash = node_id;
        }
        self.node_cnt += 1;

        let nd = &mut self.node[node_id];
        nd.clear();
        nd.move_cnt = b_info.1;
        nd.hash = hs;
        nd.init_branch();

        for &rv in &np::argsort(prob, true) {
            if b_info.2.contains(&rv) {
                nd.mov[nd.branch_cnt] = rv2ev(rv);
                nd.prob[nd.branch_cnt] = prob[rv];
                nd.branch_cnt += 1;
            }
        }

        node_id
    }

    /// node_idのノードの先を探索し、ValueNetworkの値を返します。
    fn search_branch(
        &mut self,
        b: &mut Board,
        node_id: usize,
        route: &mut Vec<(usize, usize)>,
    ) -> f32 {
        let best;
        let next_id;
        let next_move;
        let is_head_node;
        {
            // 上記変数を計算し、下記ndを解放するためのブロック
            use itertools::multizip;

            let nd = &self.node[node_id];
            let nd_rate = if nd.total_cnt == 0 {
                0.0
            } else {
                nd.total_value / nd.total_cnt as f32
            };
            let cpsv = unsafe { TREE_CP } * (nd.total_cnt as f32).sqrt();
            let rate = nd.value_win
                .iter()
                .zip(nd.visit_cnt.iter())
                .map(|(&w, &c)| if c == 0 { nd_rate } else { w / c as f32 });
            let action_value = multizip((rate, nd.prob.iter(), nd.visit_cnt.iter()))
                .map(|(r, &p, &c)| r + cpsv * p / (c + 1) as f32)
                .take(nd.branch_cnt);
            best = np::argmax(action_value);
            next_id = nd.next_id[best];
            next_move = nd.mov[best];
            is_head_node = !self.has_next(node_id, best, b.get_move_cnt() + 1)
                || nd.visit_cnt[best] < EXPAND_CNT
                || b.get_move_cnt() > BVCNT * 2
                || (next_move == PASS && b.get_prev_move() == PASS);
        }
        route.push((node_id, best));

        let _ = b.play(next_move, false);

        let value = if is_head_node {
            if self.node[node_id].evaluated[best] {
                self.node[node_id].value[best]
            } else {
                let (prob_, value_) = self.evaluate(b);
                self.eval_cnt += 1;
                let value = -value_[0];
                {
                    let mut nd = &mut self.node[node_id];
                    nd.value[best] = value;
                    nd.evaluated[best] = true;
                }

                if self.node_cnt > (0.85 * MAX_NODE_CNT as f32) as usize {
                    self.delete_node();
                }

                let next_id = self.create_node(b.info(), &prob_);

                {
                    let nd = &mut self.node[node_id];
                    nd.next_id[best] = next_id;
                    nd.next_hash[best] = b.hash();

                    nd.total_value -= nd.value_win[best];
                    nd.total_cnt += nd.visit_cnt[best];
                }
                value
            }
        } else {
            -self.search_branch(b, next_id, route)
        };

        let nd = &mut self.node[node_id];
        nd.total_value += value;
        nd.total_cnt += 1;
        nd.value_win[best] += value;
        nd.visit_cnt[best] += 1;

        value
    }

    /// time_で決定される時間の間、MCTSを実行し、最も勝率の高い着手と勝率を返します。
    pub fn search(&mut self, b: &Board, time_: f32, ponder: bool, clean: bool) -> (usize, f32) {
        let mut time_ = time_;
        let start = time::SystemTime::now();
        let (prob, _) = self.evaluate(b);
        self.root_id = self.create_node(b.info(), &prob);
        self.root_move_cnt = b.get_move_cnt();
        unsafe {
            TREE_CP = if b.get_move_cnt() < 8 { 0.01 } else { 1.5 };
        }

        if self.node[self.root_id].branch_cnt <= 1 {
            eprintln!("\nmove count={}:", b.get_move_cnt() + 1);
            self.print_info(self.root_id);
            return (PASS, 0.5);
        }

        self.delete_node();

        let mut best;
        let mut second;
        let stand_out;
        let almost_win;
        {
            let nd = &self.node[self.root_id];

            let order_ = np::argsort(&nd.visit_cnt[0..nd.branch_cnt], true);
            best = order_[0];
            second = order_[1];

            let win_rate = self.branch_rate(nd, best);

            stand_out = nd.total_cnt > 5000 && nd.visit_cnt[best] > nd.visit_cnt[second] * 100;
            almost_win = nd.total_cnt > 5000 && (win_rate < 0.1 || win_rate > 0.9);
        }

        if ponder || !(stand_out || almost_win) {
            if time_ == 0.0 {
                if self.main_time == 0.0 || self.left_time < self.byoyomi * 2.0 {
                    time_ = self.byoyomi.max(1.0);
                } else {
                    time_ = self.left_time / (55.0 + (50 - b.get_move_cnt()).max(0) as f32);
                }
            }
            // search
            let mut search_idx = 1;
            self.eval_cnt = 0;
            let mut b_cpy = Board::new();
            loop {
                b.copy_to(&mut b_cpy);
                let mut route = Vec::new();
                let root_id = self.root_id;
                self.search_branch(&mut b_cpy, root_id, &mut route);
                search_idx += 1;
                if search_idx % 64 == 0 && (ponder && TREE_STOP.load(Ordering::Relaxed))
                    || duration2float(start.elapsed().unwrap()) > time_
                {
                    TREE_STOP.store(false, Ordering::Relaxed);
                    break;
                }
            }
            let nd = &self.node[self.root_id];
            let order_ = np::argsort(&nd.visit_cnt[0..nd.branch_cnt], true);
            best = order_[0];
            second = order_[1];
        }

        let nd = &self.node[self.root_id];
        let mut next_move = nd.mov[best];
        let mut win_rate = self.branch_rate(&nd, best);

        if clean && next_move == PASS && nd.value_win[best] * nd.value_win[second] > 0.0 {
            next_move = nd.mov[second];
            win_rate = self.branch_rate(&nd, second);
        }
        if !ponder {
            eprintln!(
                "\nmove count={}: left time={:.1}[sec] evaluated={}",
                b.get_move_cnt() + 1,
                (self.left_time - time_).max(0.0),
                self.eval_cnt
            );
            self.print_info(self.root_id);
            self.left_time = (self.left_time - duration2float(start.elapsed().unwrap())).max(0.0);
        }

        (next_move, win_rate)
    }

    fn has_next(&self, node_id: usize, br_id: usize, move_cnt: usize) -> bool {
        let nd = &self.node[node_id];
        let next_id = nd.next_id[br_id];
        next_id < usize::max_value() && nd.next_hash[br_id] == self.node[next_id].hash
            && self.node[next_id].move_cnt == move_cnt
    }

    fn branch_rate(&self, nd: &Node, id: usize) -> f32 {
        nd.value_win[id] / nd.visit_cnt[id].max(1) as f32 / 2.0 + 0.5
    }

    fn best_sequence(&self, node_id: usize, head_move: usize) -> String {
        let mut node_id = node_id;
        let mut seq_str = format!("{:>3}", ev2str(head_move));
        let mut next_move = head_move;

        for _ in 0..7 {
            let nd = &self.node[node_id];
            if next_move == PASS || nd.branch_cnt < 1 {
                break;
            }

            let best = np::argmax(nd.visit_cnt[0..nd.branch_cnt].iter());
            if nd.visit_cnt[best] == 0 {
                break;
            }
            next_move = nd.mov[best];
            seq_str = format!("{}->{:>3}", seq_str, ev2str(next_move));

            if !self.has_next(node_id, best, nd.move_cnt + 1) {
                break;
            }
            node_id = nd.next_id[best];
        }

        seq_str
    }

    pub fn print_info(&self, node_id: usize) {
        let nd = &self.node[node_id];
        let order_ = np::argsort(&nd.visit_cnt[0..nd.branch_cnt], true);
        eprintln!("|move|count  |rate |value|prob | best sequence");
        for i in 0..order_.len().min(9) {
            let m = order_[i];
            let visit_cnt = nd.visit_cnt[m];
            if visit_cnt == 0 {
                break;
            }

            let rate = if visit_cnt == 0 {
                0.0
            } else {
                self.branch_rate(&nd, m) * 100.0
            };
            let value = (nd.value[m] / 2.0 + 0.5) * 100.0;

            eprintln!(
                "|{:>4}|{:7}|{:5.1}|{:5.1}|{:5.1}| {}",
                ev2str(nd.mov[m]),
                visit_cnt,
                rate,
                value,
                nd.prob[m] * 100.0,
                self.best_sequence(nd.next_id[m], nd.mov[m])
            );
        }
    }
}
