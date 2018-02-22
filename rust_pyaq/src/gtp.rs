use std::vec::Vec;
use std::io::{self, BufRead};
use numpy as np;
use constants::*;
use board::*;
use search::Tree;


const CMD_LIST: [&str; 15] = [
    "protocol_version",
    "name",
    "version",
    "list_commands",
    "boardsize",
    "komi",
    "time_settings",
    "time_left",
    "clear_board",
    "genmove",
    "play",
    "undo",
    "gogui-play_sequence",
    "showboard",
    "quit",
];


fn send(res_cmd: &str) {
    println!("= {}\n", res_cmd);
}


fn parse(line: &str) -> (Option<&str>, Vec<&str>) {
    let mut args = line.split_whitespace();
    let command = if let Some(first) = args.next() {
        if first == "=" {
            args.next()
        } else {
            Some(first)
        }
    } else {
        None
    };
    (command, args.collect())
}


/// GTPコマンドを待ち受け、実行するループです。
pub fn call_gtp(main_time: f32, byoyomi: f32, quick: bool, clean: bool) {
    let mut b = Board::new();
    let mut tree = Tree::new("frozen_model.pb");
    tree.set_time(main_time, byoyomi);

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let line = line.trim_right();
        if line.is_empty() {
            continue;
        }
        let (command, args) = parse(line);
        match command.unwrap() {
            "protocol_version" => { send("2"); },
            "name" => { send("AlphaGo9"); },
            "version" => { send("1.0"); },
            "list_commands" => {
                print!("=");
                for cmd in CMD_LIST.iter() {
                    println!("{}", cmd);
                }
                send("");
            },
            "boardsize" => {
                if let Some(arg) = args.get(0) {
                    let bs = arg.parse::<usize>().unwrap();
                    if bs == BSIZE {
                        send("");
                    } else {
                        println!("?invalid boardsize\n");
                    }
                } else {
                    println!("?invalid boardsize\n");
                }
            },
            "komi" => {
                if let Some(arg) = args.get(0) {
                    let bs = arg.parse::<f32>().unwrap();
                    if bs == KOMI {
                        send("");
                    } else {
                        println!("?invalid komi\n");
                    }
                } else {
                    println!("?invalid komi\n");
                }
            },
            "time_settings" => {
                tree.set_time(args[0].parse().unwrap(), args[1].parse().unwrap());
                send("");
            },
            "time_left" => {
                tree.set_left_time(args[1].parse().unwrap());
                send("");
            },
            "clear_board" => {
                b.clear();
                tree.clear();
                send("");
            },
            "genmove" => {
                let (mov, win_rate) = if quick {
                    (rv2ev(np::argmax(&tree.evaluate(&b).unwrap().0)), 0.5)
                } else {
                    tree.search(&b, 0.0, false, clean)
                };
                if win_rate < 0.1 {
                    send("resign");
                } else {
                    let _ = b.play(mov, true);
                    send(&ev2str(mov));
                }
            },
            "play" => {
                let _ = b.play(str2ev(args[1]), false);
                send("");
            },
            "undo" => {
                let mut history = b.get_history().clone();
                history.pop();
                b.clear();
                tree.clear();
                for v in history {
                    let _ = b.play(v, false);
                }
                send("");
            },
            "gogui-play_sequence" => {
                let mut a = args.iter();
                while let Some(_) = a.next() {
                    if let Some(mov) = a.next() {
                        let _ = b.play(str2ev(mov), false);
                    } else {
                        break;
                    }
                }
                send("");
            },
            "showboard" => {
                b.showboard();
                send("");
            },
            "quit" => {
                send("");
                break;
            },
            _ => {
                println!("?unknown_command\n");
            },
        }
    }
}
