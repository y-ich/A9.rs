use std::fs::File;
use std::io::Read;
use std::error::Error;
use tensorflow as tf;

pub struct NeuralNetwork {
    graph: tf::Graph,
    session: tf::Session,
}

impl NeuralNetwork {
    pub fn new(name: &str) -> Self {
        let mut graph = tf::Graph::new();
        let mut proto = Vec::new();
        File::open(name).unwrap().read_to_end(&mut proto).unwrap();
        graph.import_graph_def(&proto, &tf::ImportGraphDefOptions::new()).unwrap();
        let session = tf::Session::new(&tf::SessionOptions::new(), &graph).unwrap();
        NeuralNetwork {
            graph: graph,
            session: session,
        }
    }

    pub fn evaluate(&mut self, feature: &tf::Tensor<f32>) -> Result<(tf::Tensor<f32>, tf::Tensor<f32>), Box<Error>> {
        let mut step = tf::StepWithGraph::new();
        step.add_input(&self.graph.operation_by_name_required("x")?, 0, &feature);
        let policy = step.request_output(&self.graph.operation_by_name_required("pfc/policy")?, 0);
        let value = step.request_output(&self.graph.operation_by_name_required("vfc/value")?, 0);
        self.session.run(&mut step)?;
        Ok((step.take_output(policy)?, step.take_output(value)?))
    }
}
