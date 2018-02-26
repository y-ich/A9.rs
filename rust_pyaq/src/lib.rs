#![feature(test)]
#![feature(box_syntax)]

extern crate test;
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate rand;

#[cfg(not(target_arch = "wasm32"))]
extern crate tensorflow;

#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate stdweb;

pub mod utils;
pub mod numpy;
pub mod constants;
pub mod stone_group;
pub mod board;
pub mod search;

#[cfg(not(target_arch = "wasm32"))]
pub mod neural_network;
