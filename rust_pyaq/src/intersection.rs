/// 石の色や手番を表す列挙型です。
#[derive(Clone, Copy, PartialEq, Hash)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub fn opponent(&self) -> Self {
        match *self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

// 交点の状態を表す列挙型です。
#[derive(Clone, Copy, PartialEq, Hash)]
pub enum Intersection {
    Stone(Color),
    Empty,
    Exterior, // 盤の外
}

impl Intersection {
    #[inline]
    pub fn to_usize(&self) -> usize {
        match *self {
            Intersection::Stone(c) => c as usize,
            Intersection::Empty => 2,
            Intersection::Exterior => 3,
        }
    }
}
