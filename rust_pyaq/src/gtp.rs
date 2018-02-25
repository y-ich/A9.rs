use std::io;
use sgf::SgfCollection;
use numpy as np;
use constants::*;
use board::*;
use search::Tree;

fn response_list_commands() {
    const CMD_LIST: [&str; 16] = [
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
        "loadsgf",
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

fn move2xy(mov: &str) -> (u8, u8) {
    const OFFSET: u8 = 'a' as u8 - 1;
    let mut chars = mov.chars();
    let first = chars.next().unwrap();
    let second = chars.next().unwrap();
    (first as u8 - OFFSET, second as u8 - OFFSET)
}

fn read_file(name: &str) -> io::Result<String> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(name)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
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
                let (mov, win_rate) = self.best_move();
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
                self.tree.clear();
                self.b.clear();
                self.b.play_sequence(history.into_iter());
                send("");
            }
            "gogui-play_sequence" => {
                let mut a = args.iter();
                a.next();
                self.b.play_sequence(a.step_by(2).map(|s| str2ev(&s)));
                send("");
            }
            "showboard" => {
                self.b.showboard();
                send("");
            }
            "loadsgf" => {
                if let Some(filename) = args.get(0) {
                    if let Ok(sgf) = read_file(filename) {
                        if let Ok(collection) = SgfCollection::from_sgf(&sgf) {
                            let mn = if let Some(mn) = args.get(1) {
                                mn.parse::<usize>().unwrap()
                            } else {
                                usize::max_value()
                            };
                            self.load_collection(&collection, mn);
                            send("");
                        } else {
                            println!("?invalid sgf\n");
                        }
                    } else {
                        println!("?cannot open file\n");
                    }
                } else {
                    println!("?missing filename\n");
                }
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

    /// sgfテキストをロードして次の手番を返します。
    pub fn load_sgf(&mut self, sgf: &str, mn: usize) -> Result<Color, &'static str> {
        if let Ok(collection) = SgfCollection::from_sgf(&sgf) {
            Ok(self.load_collection(&collection, mn))
        } else {
            Err("invalid sgf")
        }
    }

    fn load_collection(&mut self, collection: &SgfCollection, mn: usize) -> Color {
        self.tree.clear();
        self.b.clear();
        // TODO - play_sequenceを使う。generatorが良さそうだけどまだnightly
        let mut node = &collection[0];
        let mut n = 0;
        let mut color = Color::Black;

        while node.children.len() > 0 {
            if n >= mn {
                break;
            }
            node = &node.children[0];
            if let Ok(point) = node.get_point("B").or(node.get_point("W")) {
                let (x, y) = move2xy(&point);
                let _ = self.b.play(xy2ev(x, y), false);
                n += 1;
            }
            // TODO - get_pointの二重コール避けたい。でもきれいに書けない。
            if node.get_point("B").is_ok() {
                color = Color::White;
            } else if node.get_point("W").is_ok() {
                color = Color::Black;
            }
        }
        color
    }

    /// 現局面の探索最善手と勝率を返します。手番はself.b.turnです。
    pub fn best_move(&mut self) -> (usize, f32) {
        if self.quick {
            (rv2ev(np::argmax(self.tree.evaluate(&self.b).0.iter())), 0.5)
        } else {
            self.tree.search(&self.b, 0.0, false, self.clean)
        }
    }
}
