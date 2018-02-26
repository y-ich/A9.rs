use intersection::*;
use board::*;
use search::{Evaluate, Tree};

/// JavaSriptコマンド用ワーカーです。
pub struct JsClient {
    b: Board,
    tree: Tree<NeuralNetwork>,
}

impl JsClient {
    pub fn new() -> Self {
        let mut tree = Tree::new(NeuralNetwork {});
        tree.set_time(0.0, 1.0);
        Self {
            b: Board::new(),
            tree: tree,
        }
    }

    pub fn load_pv(&mut self, pv: &[usize]) -> Result<Color, Error> {
        self.tree.clear();
        self.b.clear();
        let mut color = Color::Black;

        for &m in pv {
            if let Err(e) = self.b.play(m, false) {
                return Err(e);
            } else {
                color = color.opponent();
            }
        }
        Ok(color)
    }

    pub fn best_move(&mut self, byoyomi: f32) -> (usize, f32) {
        self.tree.search(&self.b, byoyomi, false, false)
    }
}

pub struct NeuralNetwork {}

impl Evaluate for NeuralNetwork {
    fn evaluate(&mut self, board: &Board) -> (Vec<f32>, Vec<f32>) {
        use std::mem;
        use stdweb::{Reference, UnsafeTypedArray};
        use stdweb::web::TypedArray;
        use stdweb::unstable::TryInto;
        use constants::*;

        let mut features: [f32; BVCNT * FEATURE_CNT] = unsafe { mem::uninitialized() };
        board.put_features(&mut features);
        let features = unsafe { UnsafeTypedArray::new(&features) };
        let array: Vec<Reference> = js! { evaluate(@{features}) }.try_into().unwrap();
        let mut iter = array
            .into_iter()
            .map(|e| e.downcast::<TypedArray<f32>>().unwrap().to_vec());
        (iter.next().unwrap(), iter.next().unwrap())
    }
}
