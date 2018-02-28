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

    pub fn best_move(&mut self, playout: usize) -> (usize, f32) {
        self.tree.search(&self.b, playout, false, false)
    }
}

pub struct NeuralNetwork {}

impl Evaluate for NeuralNetwork {
    fn evaluate(&mut self, board: &Board) -> (Vec<f32>, Vec<f32>) {
        use std::mem;
        use futures::Future;
        use futures::sync::oneshot;
        use stdweb::{Promise, PromiseFuture, Reference, UnsafeTypedArray};
        use stdweb::web::TypedArray;
        use stdweb::web::error::Error;
        use stdweb::unstable::TryInto;
        use constants::*;

        let mut features: [f32; BVCNT * FEATURE_CNT] = unsafe { mem::uninitialized() };
        board.put_features(&mut features);
        let features = unsafe { UnsafeTypedArray::new(&features) };
        let jsPromise = js! { return evaluate(@{ features }) }.try_into().unwrap();
        let promise = Promise::from_thenable(&jsPromise).unwrap();
        let future: PromiseFuture<Vec<Reference>, Error> = promise.to_future();
        PromiseFuture::spawn(
            future
                .map(|array| {
                    let mut iter = array
                        .into_iter()
                        .map(|e| e.downcast::<TypedArray<f32>>().unwrap().to_vec());
                    unimplemented!();
                    // TODO - spawnされたスレッドから以下のデータをevaluateの呼び出し元に返す方法がわからない
                    (iter.next().unwrap(), iter.next().unwrap())
                })
                .map_err(|e| {
                    console!(error, e);
                }),
        );
        unimplemented!();
    }
}
