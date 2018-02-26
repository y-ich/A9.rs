use intersection::*;
use board::*;
use search::Tree;

/// JavaSriptコマンド用ワーカーです。
pub struct JsClient {
    b: Board,
    tree: Tree,
}

impl JsClient {
    pub fn new() -> Self {
        let mut tree = Tree::new();
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
