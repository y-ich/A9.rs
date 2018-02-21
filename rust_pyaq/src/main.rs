#![feature(box_syntax)]
extern crate getopts;
extern crate rand;
extern crate itertools;
extern crate tensorflow;

mod utils;
mod numpy;
mod gtp;
mod board;
mod model;
mod search;

use board::*;

enum LaunchMode {
    Gtp,
    SelfPlay,
}

fn print_usage(program: &str, opts: getopts::Options) {
    let brief = format!("Usage: {} [options]", program);
    print!("{}", opts.usage(&brief));
}

fn make_opts() -> getopts::Options {
    let mut opts = getopts::Options::new();
    opts.optflag("h", "help", "Help")
        .optflag("", "self", "Self play.")
        .optflag("", "quick", "No MCTS.")
        .optflag("", "random", "Random play.")
        .optflag("", "cpu", "CPU only, no GPUs.")
        .optflag("", "clean", "clean.")
        .optopt("", "main_time", "Main time", "NUM")
        .optopt("", "byoyomi", "Byoyomi", "NUM");
    opts
}

fn main() {
    let opts = make_opts();
    let args: Vec<String> = std::env::args().collect();
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => { m }
        Err(f) => { panic!(f.to_string()) }
    };
    if matches.opt_present("h") {
        print_usage(&args[0], opts);
        std::process::exit(0);
    }
    let launch_mode = if matches.opt_present("self") {
        LaunchMode::SelfPlay
    } else {
        LaunchMode::Gtp
    };
    let quick = matches.opt_present("quick");
    let random = matches.opt_present("random");
    let clean = matches.opt_present("clean");
    let main_time = matches.opt_str("main_time").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let byoyomi = matches.opt_str("byoyomi").and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let use_gpu = !matches.opt_present("cpu");

    match launch_mode {
        LaunchMode::Gtp => {
            gtp::call_gtp(main_time, byoyomi, quick, clean, use_gpu);
        },
        LaunchMode::SelfPlay => {
            let mut b = Board::new();
            if random {
                while b.move_cnt < BVCNT * 2 {
                    let prev_move = b.prev_move;
                    let mov = b.random_play();
                    let _ = b.play(mov, false);
                    b.showboard();
                    if prev_move == PASS && mov == PASS {
                        break;
                    }
                }
            } else {
                let mut tree = search::Tree::new("frozen_model.pb", use_gpu);
                while b.move_cnt < BVCNT * 2 {
                    let prev_move = b.prev_move;
                    let (mov, _) = tree.search(&b, 0.0, false, clean);
                    let _ = b.play(mov, false);
                    b.showboard();
                    if prev_move == PASS && mov == PASS {
                        break;
                    }
                }
            }

            let mut score_list = Vec::new();
            let mut b_cpy = Board::new();

            for _ in 0..256 {
                b.copy_to(&mut b_cpy);
                b_cpy.rollout(false);
                score_list.push(b_cpy.score());
            }

            let score = utils::most_common(&score_list);
            let result_str = if score == 0.0 {
                "Draw".to_string()
            } else {
                let winner = if score > 0.0 { "B" } else { "W" };
                format!("{}+{:.1}", winner, score.abs())
            };
            eprintln!("result: {}", result_str);
        },
    }
}
