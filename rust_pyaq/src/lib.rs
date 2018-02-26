#![feature(box_syntax)]

extern crate itertools;
#[cfg(feature = "ponder")]
#[warn(unused_imports)]
#[macro_use]
extern crate lazy_static;
extern crate rand;

pub mod utils;
pub mod numpy;
pub mod constants;
pub mod intersection;
pub mod coord_convert;
pub mod stone_group;
pub mod board;
pub mod search;
