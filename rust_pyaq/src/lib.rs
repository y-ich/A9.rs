#![feature(box_syntax)]
#![feature(iterator_step_by)]
#![feature(proc_macro)] // stdwebが使う
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate rand;
#[cfg(not(target_arch = "wasm32"))]
extern crate sgf; // libcを使うのでwasm32では使わないようにする
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
#[cfg(not(target_arch = "wasm32"))]
pub mod gtp;
#[cfg(target_arch = "wasm32")]
mod js_client;

#[cfg(target_arch = "wasm32")]
#[js_export]
pub fn think(pv: Vec<(usize)>, byoyomi: f32) -> (usize, f32) {
    use std::f32;

    let mut client = js_client::JsClient::new();
    if client.load_pv(&pv).is_ok() {
        client.best_move(byoyomi)
    } else {
        (usize::max_value(), f32::NAN)
    }
}
