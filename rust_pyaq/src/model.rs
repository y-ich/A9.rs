use std::fs::File;
use std::io::Read;
use std::error::Error;
use tensorflow as tf;

pub struct DualNetwork {}

impl DualNetwork {
    pub fn new() -> Self { DualNetwork {} }

    pub fn create_sess(&mut self, name: &str) -> Result<(tf::Graph, tf::Session), Box<Error>> {
        let mut graph = tf::Graph::new();
        let mut proto = Vec::new();
        File::open(name)?.read_to_end(&mut proto)?;
        graph.import_graph_def(&proto, &tf::ImportGraphDefOptions::new())?;
        let session = tf::Session::new(&tf::SessionOptions::new(), &graph).unwrap();
        Ok((graph, session))
    }

}
