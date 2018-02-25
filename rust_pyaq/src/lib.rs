#![feature(box_syntax)]
#![feature(iterator_step_by)]
#![feature(proc_macro)] // stdwebが使う
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate sgf;
#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate stdweb;
#[cfg(not(target_arch = "wasm32"))]
extern crate tensorflow;

pub mod utils;
pub mod numpy;
pub mod constants;
pub mod stone_group;
pub mod board;
#[cfg(not(target_arch = "wasm32"))]
pub mod neural_network;
pub mod search;
pub mod gtp;

#[cfg(target_arch = "wasm32")]
#[js_export]
pub fn think(sgf: &str, byoyomi: f32) -> ((u8, u8), f32) {
    use std::f32;

    let mut gtp = gtp::GtpClient::new(0.0, byoyomi, false, false);
    if gtp.load_sgf(sgf, usize::max_value()).is_ok() {
        let (mov, win_rate) = gtp.best_move();
        (board::ev2xy(mov), win_rate)
    } else {
        ((u8::max_value(), u8::max_value()), f32::NAN)
    }
}
