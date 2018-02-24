#![feature(box_syntax)]
#![feature(proc_macro)] // stdwebが使う
extern crate itertools;
extern crate rand;
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
