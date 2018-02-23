#![feature(box_syntax)]
extern crate rand;
extern crate itertools;
#[cfg(not(feature = "wasm"))]
extern crate tensorflow;
#[cfg(feature = "wasm")]
extern crate stdweb;

pub mod utils;
pub mod numpy;
pub mod constants;
pub mod stone_group;
pub mod board;
#[cfg(not(feature = "wasm"))]
pub mod neural_network;
pub mod search;
pub mod gtp;

// wasm化のためにexport
pub use::gtp::gtp;
