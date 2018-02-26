#![feature(iterator_step_by)]
/// rust_pyaq: Pyaq(https://github.com/ymgaq/Pyaq)のRustへの移植コード
/// 作者: 市川雄二
/// ライセンス: MIT
extern crate getopts;
extern crate rust_pyaq_lib;
extern crate sgf;
pub mod gtp;

use rust_pyaq_lib as rpl;
use rpl::*;
use rpl::constants::*;
use rpl::board::*;

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
        // .optflag("", "cpu", "CPU only, no GPUs.") // 現状、
        .optflag("", "clean", "Try to pickup all dead stones.")
        .optopt("", "main_time", "Main time(sec) defaut: 0", "NUM")
        .optopt("", "byoyomi", "Byoyomi(sec) default: 3 (1 for self play)", "NUM");
    opts
}

fn random_self_play(max_move_cnt: usize) -> Board {
    let mut b = Board::new();
    while b.get_move_cnt() < max_move_cnt {
        let prev_move = b.get_prev_move();
        let mov = b.random_play();
        let _ = b.play(mov, false);
        b.showboard();
        if prev_move == PASS && mov == PASS {
            break;
        }
    }
    b
}

fn self_play(max_move_cnt: usize, time: f32, clean: bool) -> Board {
    use neural_network::NeuralNetwork;

    let mut b = Board::new();
    let mut tree = search::Tree::new(NeuralNetwork::new("frozen_model.pb"));
    while b.get_move_cnt() < max_move_cnt {
        let prev_move = b.get_prev_move();
        let (mov, _) = tree.search(&b, time, false, clean);
        let _ = b.play(mov, false);
        b.showboard();
        if prev_move == PASS && mov == PASS {
            break;
        }
    }
    b
}

fn main() {
    let opts = make_opts();
    let args: Vec<String> = std::env::args().collect();
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!(f.to_string()),
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
    let main_time = matches
        .opt_str("main_time")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let byoyomi = matches
        .opt_str("byoyomi")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    // let use_gpu = !matches.opt_present("cpu");

    match launch_mode {
        LaunchMode::Gtp => {
            gtp::GtpClient::new(main_time, byoyomi, quick, clean).call_gtp();
        }
        LaunchMode::SelfPlay => {
            let end_position = if random {
                random_self_play(BVCNT * 2)
            } else {
                self_play(BVCNT * 2, 0.0, clean)
            };

            let score = end_position.final_score();
            let result_str = if score == 0.0 {
                "Draw".to_string()
            } else {
                let winner = if score > 0.0 { "B" } else { "W" };
                format!("{}+{:.1}", winner, score.abs())
            };
            eprintln!("result: {}", result_str);
        }
    }
}
