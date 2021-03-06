#![feature(proc_macro)] // stdwebが使う

extern crate futures;
extern crate rust_pyaq;
extern crate serde;
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate stdweb;

mod js_client;

use stdweb::js_export;
use rust_pyaq::*;

#[derive(Serialize)]
pub struct MoveInfo {
    mov: usize,
    win_rate: f64,
}

js_serializable!(MoveInfo);

#[js_export]
pub fn think(pv: Vec<usize>, playout: usize) -> MoveInfo {
    use std::f64;

    let mut client = js_client::JsClient::new();
    if client.load_pv(&pv).is_ok() {
        let result = client.best_move(playout);
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
