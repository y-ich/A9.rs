use std::usize;
use std::cmp::{min, max};
use std::mem;
use std::error::Error;
use std::collections::HashMap;
use std::time;
use itertools::multizip;
use tensorflow as tf;
use utils::fill;
use numpy as np;
use board::*;
use model::DualNetwork;

const MAX_NODE_CNT: usize = 16384; // 2 ^ 14
const EXPAND_CNT: usize = 8;

fn duration2float(d: time::Duration) -> f32 {
    d.as_secs() as f32 + d.subsec_nanos() as f32 / 1000_000_000.0
}

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
    move_cnt: usize,
}

impl Node {
    pub fn new() -> Self {
        let mut node: Self = unsafe { mem::uninitialized() };
        node.init_branch();
        node.clear();
        node
    }

    pub fn init_branch(&mut self) {
        fill(&mut self.mov, VNULL);
        fill(&mut self.prob, 0.0);
        fill(&mut self.value, 0.0);
        fill(&mut self.value_win, 0.0);
        fill(&mut self.visit_cnt, 0);
        fill(&mut self.next_id, usize::MAX);
        fill(&mut self.next_hash, 0);
        fill(&mut self.evaluated, false);
    }

    pub fn clear(&mut self) {
        self.branch_cnt = 0;
        self.total_value = 0.0;
        self.total_cnt = 0;
        self.hash = 0;
        self.move_cnt = usize::MAX;
    }
}


static mut TREE_STOP: bool = false;
static mut TREE_CP: f32 = 2.0;

pub struct Tree {
    pub main_time: f32,
    pub byoyomi: f32,
    pub left_time: f32,
    node: Box<[Node; MAX_NODE_CNT]>,
    node_cnt: usize,
    root_id: usize,
    root_move_cnt: usize,
    node_hashs: HashMap<u64, usize>,
    eval_cnt: usize,
    graph: tf::Graph,
    sess: tf::Session,
}

impl Tree {
    pub fn new(ckpt_path: &str, use_gpu: bool) -> Self {
        let (graph, sess) = Self::get_sess(ckpt_path, use_gpu);
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
            graph: graph,
            sess: sess,
        }
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
        unsafe { TREE_STOP = false; }
    }

    pub fn get_sess(ckpt_path: &str, use_gpu: bool) -> (tf::Graph, tf::Session) {
        let _device_name = if use_gpu { "gpu" } else { "cpu" };
        let mut dn = DualNetwork::new();
        dn.create_sess(ckpt_path).unwrap()
    }

    pub fn evaluate(&mut self, b: &Board) -> Result<(tf::Tensor<f32>, tf::Tensor<f32>), Box<Error>>{
        let feature = b.feature();
        let mut step = tf::StepWithGraph::new();
        step.add_input(&self.graph.operation_by_name_required("x")?, 0, &feature);
        let policy = step.request_output(&self.graph.operation_by_name_required("pfc/policy")?, 0);
        let value = step.request_output(&self.graph.operation_by_name_required("vfc/value")?, 0);
        self.sess.run(&mut step)?;
        Ok((step.take_output(policy)?, step.take_output(value)?))
    }

    pub fn delete_node(&mut self) {
        if self.node_cnt < MAX_NODE_CNT / 2 {
            return;
        }
        for i in 0..MAX_NODE_CNT {
            let mc = self.node[i].move_cnt;
            if mc < usize::MAX && mc < self.root_move_cnt {
                if self.node_hashs.contains_key(&self.node[i].hash)  {
                    self.node_hashs.remove(&self.node[i].hash);
                }
                self.node[i].clear()
            }
        }
    }

    pub fn create_node(&mut self, b_info: (u64, usize, Vec<usize>), prob: &[f32]) -> usize {
        let hs = b_info.0;

        if self.node_hashs.contains_key(&hs) &&
            self.node[self.node_hashs[&hs]].hash == hs &&
            self.node[self.node_hashs[&hs]].move_cnt == b_info.1 {
            return self.node_hashs[&hs];
        }

        let mut node_id = hs as usize % MAX_NODE_CNT;

        while self.node[node_id].move_cnt != usize::MAX {
            node_id = if node_id + 1 < MAX_NODE_CNT { node_id + 1 } else { 0 };
        }

        if let Some(hash) = self.node_hashs.get_mut(&hs) {
            *hash = node_id;
        }
        self.node_cnt += 1;

        let nd = self.node.get_mut(node_id).unwrap();
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

        return node_id;
    }

    pub fn search_branch(&mut self, b: &mut Board, node_id: usize, route: &mut Vec<(usize, usize)>) -> f32 {
        let best;
        let next_id;
        let next_move;
        let head_node;
        { // ndを解放するためのブロック
            let nd = self.node.get(node_id).unwrap();
            let nd_rate = if nd.total_cnt == 0 { 0.0 } else { nd.total_value / nd.total_cnt as f32 };
            let cpsv = unsafe { TREE_CP } * (nd.total_cnt as f32).sqrt();
            let rate: Vec<f32> = nd.value_win.iter().zip(nd.visit_cnt.iter())
                .map(|(&w, &c)| if c == 0 { nd_rate } else { w / c as f32 })
                .collect();
            let action_value: Vec<f32> = multizip((rate.iter(), nd.prob.iter(), nd.visit_cnt.iter()))
                .map(|(&r, &p, &c)| r + cpsv * p / (c + 1) as f32)
                .collect();
            best = np::argmax(&action_value[0..nd.branch_cnt]);

            route.push((node_id, best));
            next_id = nd.next_id[best];
            next_move = nd.mov[best];
            head_node = !self.has_next(node_id, best, b.move_cnt + 1) ||
                nd.visit_cnt[best] < EXPAND_CNT ||
                (b.move_cnt > BVCNT * 2) ||
                (next_move == PASS && b.prev_move == PASS);
        }

        let _ = b.play(next_move, false);

        let value;
        if head_node {
            if self.node[node_id].evaluated[best] {
                value = self.node[node_id].value[best];
            } else {
                let (prob_, value_) = self.evaluate(b).unwrap();
                self.eval_cnt += 1;
                value = -value_[0];
                self.node[node_id].value[best] = value;
                self.node[node_id].evaluated[best] = true;

                if self.node_cnt as f32 > 0.85 * MAX_NODE_CNT as f32 {
                    self.delete_node();
                }

                let next_id = self.create_node(b.info(), &prob_);

                self.node[node_id].next_id[best] = next_id;
                self.node[node_id].next_hash[best] = b.hash();

                self.node[next_id].total_value -= self.node[node_id].value_win[best];
                self.node[next_id].total_cnt += self.node[node_id].visit_cnt[best];
            }
        } else {
            value = -self.search_branch(b, next_id, route);
        }

        let nd = self.node.get_mut(node_id).unwrap();
        nd.total_value += value;
        nd.total_cnt += 1;
        nd.value_win[best] += value;
        nd.visit_cnt[best] += 1;
        return value;
    }

    pub fn search(&mut self, b: &Board, time_: f32, ponder: bool, clean: bool) -> (usize, f32) {
        let mut time_ = time_;
        let start = time::SystemTime::now();
        let (prob, _) = self.evaluate(b).unwrap();
        self.root_id = self.create_node(b.info(), &prob);
        self.root_move_cnt = b.move_cnt;
        unsafe { TREE_CP = if b.move_cnt < 8 { 0.01 } else { 1.5 }; }

        if self.node[self.root_id].branch_cnt <= 1 {
            eprintln!("\nmove count={}:", b.move_cnt + 1);
            self.print_info(self.root_id);
            return (PASS, 0.5);
        }

        self.delete_node();

        let mut order_ = np::argsort(&self.node[self.root_id].visit_cnt[0..self.node[self.root_id].branch_cnt], true);
        let mut best = order_[0];
        let mut second = order_[1];

        let mut win_rate = self.branch_rate(&self.node[self.root_id], best);

//         if not ponder and self.byoyomi == 0 and self.left_time < 10:
//         if nd.visit_cnt[best] < 1000:
//                 return rv2ev(np.argmax(prob)), 0.5
//             else:
//                 stderr.write("\nmove count=%d:\n" % (b.move_cnt + 1))
//                 self.print_info(self.root_id)
//                 return nd.move[best], win_rate

        let stand_out = self.node[self.root_id].total_cnt > 5000 && self.node[self.root_id].visit_cnt[best] > self.node[self.root_id].visit_cnt[second] * 100;
        let almost_win = self.node[self.root_id].total_cnt > 5000 && (win_rate < 0.1 || win_rate > 0.9);

        if ponder || !(stand_out || almost_win) {
            if time_ == 0.0 {
                if self.main_time == 0.0 || self.left_time < self.byoyomi * 2.0 {
                    time_ = self.byoyomi.max(1.0);
                } else {
                    time_ = self.left_time / (55.0 + max(50 - b.move_cnt, 0) as f32);
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
                if search_idx % 64 == 0 {
                    if (ponder && unsafe { TREE_STOP }) || duration2float(start.elapsed().unwrap()) > time_ {
                        unsafe { TREE_STOP = false; }
                        break;
                    }
                }
            }
            order_ = np::argsort(&self.node[self.root_id].visit_cnt[0..self.node[self.root_id].branch_cnt], true);
            best = order_[0];
            second = order_[1];
        }

        let mut next_move = self.node[self.root_id].mov[best];
        win_rate = self.branch_rate(&self.node[self.root_id], best);

        if clean && next_move == PASS {
            if self.node[self.root_id].value_win[best] * self.node[self.root_id].value_win[second] > 0.0 {
                next_move = self.node[self.root_id].mov[second];
                win_rate = self.branch_rate(&self.node[self.root_id], second);
            }
        }
        if !ponder {
            eprintln!("\nmove count={}: left time={:.1}[sec] evaluated={}",
                b.move_cnt + 1, (self.left_time - time_).max(0.0), self.eval_cnt);
            self.print_info(self.root_id);
            self.left_time = (self.left_time - duration2float(start.elapsed().unwrap())).max(0.0);
        }

        return (next_move, win_rate);
    }

    fn has_next(&self, node_id: usize, br_id: usize, move_cnt: usize) -> bool {
        let nd = self.node.get(node_id).unwrap();
        let next_id = nd.next_id[br_id];
        next_id < usize::MAX &&
            nd.next_hash[br_id] == self.node[next_id].hash &&
            self.node[next_id].move_cnt == move_cnt
    }

    fn branch_rate(&self, nd: &Node, id: usize) -> f32 {
        nd.value_win[id] / max(nd.visit_cnt[id], 1) as f32 / 2.0 + 0.5
    }

    fn best_sequence(&self, node_id: usize, head_move: usize) -> String {
        let mut node_id = node_id;
        let mut seq_str = format!("{:>3}", ev2str(head_move));
        let mut next_move = head_move;

        for _ in 0..7 {
            let nd = self.node.get(node_id).unwrap();
            if next_move == PASS || nd.branch_cnt < 1 {
                break;
            }

            let best = np::argmax(&nd.visit_cnt[0..nd.branch_cnt]);
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
        let nd = self.node.get(node_id).unwrap();
        let order_ = np::argsort(&nd.visit_cnt[0..nd.branch_cnt], true);
        eprintln!("|move|count  |rate |value|prob | best sequence");
        for i in 0..min(order_.len(), 9) {
            let m = order_[i];
            let visit_cnt = nd.visit_cnt[m];
            if visit_cnt == 0 {
                break;
            }

            let rate = if visit_cnt == 0 { 0.0 } else { self.branch_rate(&nd, m) * 100.0 };
            let value = (nd.value[m] / 2.0 + 0.5) * 100.0;

            eprintln!("|{:>4}|{:7}|{:5.1}|{:5.1}|{:5.1}| {}",
                ev2str(nd.mov[m]), visit_cnt, rate, value, nd.prob[m] * 100.0,
                self.best_sequence(nd.next_id[m], nd.mov[m]));
        }
    }
}
