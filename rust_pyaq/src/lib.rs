#![feature(box_syntax)]
/// WASM化のためのファイル

extern crate rand;
extern crate itertools;
#[cfg(not(feature = "wasm"))]
extern crate tensorflow;

pub mod utils;
pub mod numpy;
pub mod constants;
pub mod stone_group;
pub mod board;
#[cfg(not(feature = "wasm"))]
pub mod neural_network;
pub mod search;
pub mod gtp;

pub use::gtp::gtp;
