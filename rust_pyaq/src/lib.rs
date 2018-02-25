#![feature(box_syntax)]
#![feature(iterator_step_by)]
#![feature(proc_macro)] // stdwebが使う

extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate rand;

#[cfg(not(target_arch = "wasm32"))]
extern crate sgf; // libcを使うのでwasm32では使わないようにする
#[cfg(not(target_arch = "wasm32"))]
extern crate tensorflow;

#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate stdweb;
#[cfg(target_arch = "wasm32")]
extern crate serde;
#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate serde_derive;

pub mod utils;
pub mod numpy;
pub mod constants;
pub mod stone_group;
pub mod board;
pub mod search;

#[cfg(not(target_arch = "wasm32"))]
pub mod neural_network;
#[cfg(not(target_arch = "wasm32"))]
pub mod gtp;

#[cfg(target_arch = "wasm32")]
mod js_client;

#[cfg(target_arch = "wasm32")]
use stdweb::js_export;

#[cfg(target_arch = "wasm32")]
#[derive(Serialize)]
pub struct MoveInfo {
    mov: usize,
    win_rate: f64,
}

#[cfg(target_arch = "wasm32")]
js_serializable!(MoveInfo);

#[cfg(target_arch = "wasm32")]
#[js_export]
pub fn think(pv: Vec<usize>, byoyomi: f64) -> MoveInfo {
    use std::f64;

    let mut client = js_client::JsClient::new();
    if client.load_pv(&pv).is_ok() {
        let result = client.best_move(byoyomi as f32);
        MoveInfo {
            mov: result.0,
            win_rate: result.1 as f64,
        }
    } else {
        MoveInfo {
            mov: usize::max_value(),
            win_rate: f64::NAN,
        }
    }
}
