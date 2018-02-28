use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time;
#[cfg(feature = "ponder")]
use std::sync::atomic::{AtomicBool, Ordering};
use numpy as np;
use constants::*;
use coord_convert::*;
use board::*;

const MAX_NODE_CNT: usize = 16384; // 2 ^ 14
const EXPAND_CNT: usize = 8;

#[cfg(not(target_arch = "wasm32"))]
fn duration2float(d: time::Duration) -> f32 {
    d.as_secs() as f32 + d.subsec_nanos() as f32 / 1000_000_000.0
}

pub trait Evaluate {
    fn evaluate(&mut self, board: &Board) -> (Vec<f32>, Vec<f32>);
}

// TODO - ponderは用意だけでまだ未実装。
#[cfg(feature = "ponder")]
lazy_static! {
    static ref TREE_STOP: AtomicBool = AtomicBool::new(false);
}

/// UCB1のCp関連の係数？
static mut TREE_CP: f32 = 2.0;

/// MCTSを実行するワーカー構造体です。
pub struct Tree<T: Evaluate> {
    main_time: f32,
    byoyomi: f32,
    left_time: f32,
    node: Box<[Node; MAX_NODE_CNT]>,
    node_cnt: usize,
    pub root_id: usize, // ベンチマークのためにpubに
    root_move_cnt: usize,
    node_hashs: HashMap<u64, usize>,
    eval_cnt: usize,
    pub nn: T,
}

impl<T: Evaluate> Tree<T> {
    pub fn new(nn: T) -> Self {
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
            nn: nn,
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
        #[cfg(feature = "ponder")]
        TREE_STOP.store(false, Ordering::Relaxed);
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

    pub fn create_node(&mut self, b: &Board, prob: &[f32]) -> usize {
        // ベンチマークのためにpubに
        let candidates = b.candidates();
        let hs = candidates.hash;

        if self.node_hashs.contains_key(&hs) && self.node[self.node_hashs[&hs]].hash == hs
            && self.node[self.node_hashs[&hs]].move_cnt == candidates.move_cnt
        {
            return self.node_hashs[&hs];
        }

        let mut node_id = hs as usize % MAX_NODE_CNT;

        // move_cntがusize::MAXのノードが必ず1つはあると仮定している。
        // delete_nodeを適宜実行しているからこれでいい？
        // MAX_CODE_CNTで抜けるようにforで書くと遅くなる。
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
        nd.move_cnt = candidates.move_cnt;
        nd.hash = hs;
        nd.init_branch();

        for &rv in &np::argsort(prob, true) {
            if candidates.list.contains(&rv) {
                nd.mov[nd.branch_cnt] = rv2ev(rv);
                nd.prob[nd.branch_cnt] = prob[rv];
                nd.branch_cnt += 1;
            }
        }

        node_id
    }

    fn best_by_action_value(&self, b: &Board, node_id: usize) -> (usize, usize, usize, bool) {
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
        let best = np::argmax(action_value);
        let next_id = nd.next_id[best];
        let next_move = nd.mov[best];
        let is_head_node = !self.has_next(node_id, best, b.get_move_cnt() + 1)
            || nd.visit_cnt[best] < EXPAND_CNT
            || b.get_move_cnt() > BVCNT * 2
            || (next_move == PASS && b.get_prev_move() == PASS);

        (best, next_id, next_move, is_head_node)
    }

    fn evaluate_child_node(&mut self, b: &Board, node_id: usize, child: usize) -> f32 {
        let (prob_, value) = self.nn.evaluate(b);
        self.eval_cnt += 1;
        let value = -value[0];
        {
            let nd = &mut self.node[node_id];
            nd.value[child] = value;
            nd.evaluated[child] = true;
        }

        if self.node_cnt > (0.85 * MAX_NODE_CNT as f32) as usize {
            self.delete_node();
        }

        let next_id = self.create_node(b, &prob_);

        {
            let nd = &mut self.node[node_id];
            nd.next_id[child] = next_id;
            nd.next_hash[child] = b.hash();

            nd.total_value -= nd.value_win[child];
            nd.total_cnt += nd.visit_cnt[child];
        }
        value
    }

    /// node_idのノードの先を探索し、ValueNetworkの値を返します。
    // ベンチマークのためにpubに
    pub fn search_branch(
        &mut self,
        b: &mut Board,
        node_id: usize,
        route: &mut Vec<(usize, usize)>,
    ) -> f32 {
        let (best, next_id, next_move, is_head_node) = self.best_by_action_value(b, node_id);
        route.push((node_id, best));

        let _ = b.play(next_move, false);

        let value = if is_head_node {
            if self.node[node_id].evaluated[best] {
                self.node[node_id].value[best]
            } else {
                self.evaluate_child_node(b, node_id, best)
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

    fn keep_playout<F: Fn(usize) -> bool>(&mut self, b: &Board, exit_condition: F) {
        let mut search_idx = 1;
        self.eval_cnt = 0;
        let mut b_cpy = Board::new();
        loop {
            b.copy_to(&mut b_cpy);
            let root_id = self.root_id; // 下の行でownershipを解決するための変数
            self.search_branch(&mut b_cpy, root_id, &mut Vec::new());
            search_idx += 1;
            #[cfg(feature = "ponder")]
            {
                if search_idx % 64 == 0
                    && (ponder && TREE_STOP.load(Ordering::Relaxed) || exit_condition(search_idx))
                {
                    TREE_STOP.store(false, Ordering::Relaxed);
                    break;
                }
            }
            #[cfg(not(feature = "ponder"))]
            {
                if search_idx % 64 == 0 && exit_condition(search_idx) {
                    break;
                }
            }
        }
    }

    /// 探索すべきか判断します。
    fn should_search(&self, best: usize, second: usize) -> bool {
        let nd = &self.node[self.root_id];
        let win_rate = self.branch_rate(nd, best);

        nd.total_cnt <= 5000
            || (nd.visit_cnt[best] <= nd.visit_cnt[second] * 100 // ベストが突出していない
                && win_rate >= 0.1 && win_rate <= 0.9) // 形勢はっきりしていない
    }

    /// time_で決定される時間の間、MCTSを実行し、最も勝率の高い着手と勝率を返します。
    #[cfg(not(target_arch = "wasm32"))]
    pub fn search(&mut self, b: &Board, time_: f32, ponder: bool, clean: bool) -> (usize, f32) {
        let mut time_ = time_;
        let start = time::SystemTime::now();
        let (prob, _) = self.nn.evaluate(b);
        self.root_id = self.create_node(b, &prob);
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

        let (mut best, mut second) = self.node[self.root_id].best2();

        if ponder || self.should_search(best, second) {
            if time_ == 0.0 {
                if self.main_time == 0.0 || self.left_time < self.byoyomi * 2.0 {
                    time_ = self.byoyomi.max(1.0);
                } else {
                    time_ = self.left_time / (55.0 + (50 - b.get_move_cnt()).max(0) as f32);
                }
            }
            self.keep_playout(b, |_| duration2float(start.elapsed().unwrap()) > time_);
            let best2 = self.node[self.root_id].best2();
            best = best2.0;
            second = best2.1;
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
                (self.left_time - time_).max(0.0), // stand_outやalmost_winの時にずれるけれども、目をつぶる。先にleft_timeを計算すればいいがそうすると、printの時間が経過時間に含まれない。
                self.eval_cnt
            );
            self.print_info(self.root_id);
            self.left_time = (self.left_time - duration2float(start.elapsed().unwrap())).max(0.0);
        }

        (next_move, win_rate)
    }

    /// MCTSをmax_playoutのプレイアウト数実行し、最も勝率の高い着手と勝率を返します。
    /// TODO - wasmのlibstdのSystemTimeのマッピングがまだ終わっていないので作成した。マッピングされたら上記メソッドに戻す
    #[cfg(target_arch = "wasm32")]
    pub fn search(
        &mut self,
        b: &Board,
        max_playout: usize,
        ponder: bool,
        clean: bool,
    ) -> (usize, f32) {
        let (prob, _) = self.nn.evaluate(b);
        self.root_id = self.create_node(b, &prob);
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

        let (mut best, mut second) = self.node[self.root_id].best2();

        if ponder || self.should_search(best, second) {
            self.keep_playout(b, |search_idx| search_idx > max_playout);
            let best2 = self.node[self.root_id].best2();
            best = best2.0;
            second = best2.1;
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
                "\nmove count={}: evaluated={}",
                b.get_move_cnt() + 1,
                self.eval_cnt
            );
            self.print_info(self.root_id);
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

    pub fn best2(&self) -> (usize, usize) {
        let order_ = np::argsort(&self.visit_cnt[0..self.branch_cnt], true);
        (order_[0], order_[1])
    }
}
