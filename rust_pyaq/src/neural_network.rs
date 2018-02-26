use std::fs::File;
use std::io::Read;
use tensorflow as tf;
use constants::*;
use board::Board;
use search::Evaluate;

/// ポリシーネットワークとバリューネットワークを併せ持つニューラルネットワークです。
pub struct NeuralNetwork {
    graph: tf::Graph,
    session: tf::Session,
}

impl NeuralNetwork {
    /// ファイル名nameのプロトコルバッファからモデルを読み込みます。
    pub fn new(name: &str) -> Self {
        let mut graph = tf::Graph::new();
        let mut proto = Vec::new();
        File::open(name).unwrap().read_to_end(&mut proto).unwrap();
        graph
            .import_graph_def(&proto, &tf::ImportGraphDefOptions::new())
            .unwrap();
        let session = tf::Session::new(&tf::SessionOptions::new(), &graph).unwrap();
        NeuralNetwork {
            graph: graph,
            session: session,
        }
    }
}

impl Evaluate for NeuralNetwork {
    /// ニューラルネットワークを評価します。
    fn evaluate(&mut self, board: &Board) -> (Vec<f32>, Vec<f32>) {
        let mut features = tf::Tensor::new(&[BVCNT as u64, FEATURE_CNT as u64]);
        board.put_features(&mut features);
        let mut step = tf::StepWithGraph::new();
        step.add_input(
            &self.graph.operation_by_name_required("x").unwrap(),
            0,
            &features,
        );
        let policy = step.request_output(
            &self.graph.operation_by_name_required("pfc/policy").unwrap(),
            0,
        );
        let value = step.request_output(
            &self.graph.operation_by_name_required("vfc/value").unwrap(),
            0,
        );
        let _ = self.session.run(&mut step);
        // TODO - シグニチャを合わせるためにわざわざVecにコピーしている。可能ならコピーは避けたい。
        (
            step.take_output(policy).unwrap().to_vec(),
            step.take_output(value).unwrap().to_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use test::Bencher;

    #[bench]
    fn bench_search_branch(b: &mut Bencher) {
        use board::Board;
        use neural_network::NeuralNetwork;
        use search::{Evaluate, Tree};

        let mut board = Board::new();
        let mut tree = Tree::new(NeuralNetwork::new("frozen_model.pb"));
        let (prob, _) = tree.nn.evaluate(&board);
        b.iter(|| {
            board.clear();
            tree.clear();
            tree.root_id = tree.create_node(board.info(), &prob);
            let mut route = Vec::new();
            let root_id = tree.root_id;
            tree.search_branch(&mut board, root_id, &mut route);
        });
    }
}
