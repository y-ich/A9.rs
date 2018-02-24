use numpy as np;
use constants::*;
use board::*;
use search::Tree;

fn response_list_commands() {
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
    print!("=");
    for cmd in CMD_LIST.iter() {
        println!("{}", cmd);
    }
    println!("");
}

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

/// GTPコマンドを待ち受け、実行するワーカーです。
pub struct GtpClient {
    b: Board,
    tree: Tree,
    quick: bool,
    clean: bool,
}

impl GtpClient {
    pub fn new(main_time: f32, byoyomi: f32, quick: bool, clean: bool) -> Self {
        let mut tree = Tree::new("frozen_model.pb");
        tree.set_time(main_time, byoyomi);
        GtpClient {
            b: Board::new(),
            tree: tree,
            quick: quick,
            clean: clean,
        }
    }

    pub fn call_gtp(&mut self) {
        use std::io::{self, BufRead};
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            if !self.gtp(&line.unwrap()) {
                break;
            }
        }
    }

    fn gtp(&mut self, line: &str) -> bool {
        let line = line.trim_right();
        if line.is_empty() {
            return true;
        }
        let (command, args) = parse(line);
        match command.unwrap() {
            "protocol_version" => {
                send("2");
            }
            "name" => {
                send("AlphaGo9");
            }
            "version" => {
                send("1.0");
            }
            "list_commands" => {
                response_list_commands();
            }
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
            }
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
            }
            "time_settings" => {
                self.tree
                    .set_time(args[0].parse().unwrap(), args[1].parse().unwrap());
                send("");
            }
            "time_left" => {
                self.tree.set_left_time(args[1].parse().unwrap());
                send("");
            }
            "clear_board" => {
                self.b.clear();
                self.tree.clear();
                send("");
            }
            "genmove" => {
                let (mov, win_rate) = if self.quick {
                    (rv2ev(np::argmax(self.tree.evaluate(&self.b).0.iter())), 0.5)
                } else {
                    self.tree.search(&self.b, 0.0, false, self.clean)
                };
                if win_rate < 0.1 {
                    send("resign");
                } else {
                    let _ = self.b.play(mov, true);
                    send(&ev2str(mov));
                }
            }
            "play" => {
                let _ = self.b.play(str2ev(args[1]), false);
                send("");
            }
            "undo" => {
                let mut history = self.b.get_history().clone();
                history.pop();
                self.b.clear();
                self.tree.clear();
                for v in history {
                    let _ = self.b.play(v, false);
                }
                send("");
            }
            "gogui-play_sequence" => {
                let mut a = args.iter();
                while let Some(_) = a.next() {
                    if let Some(mov) = a.next() {
                        let _ = self.b.play(str2ev(mov), false);
                    } else {
                        break;
                    }
                }
                send("");
            }
            "showboard" => {
                self.b.showboard();
                send("");
            }
            "quit" => {
                send("");
                return false;
            }
            _ => {
                println!("?unknown_command\n");
            }
        }
        return true;
    }
}
