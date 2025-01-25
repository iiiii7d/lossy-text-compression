use std::env::args;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::Dataset,
    },
    nn::{
        loss::{CrossEntropyLoss, CrossEntropyLossConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, CpuUse, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};

const CHARS: &str = " !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n";
const CHUNK_SIZE: usize = 64;
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    activation: Relu,
    dropout: Dropout,
    loss: CrossEntropyLoss<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3, Int>) -> Tensor<B, 2> {
        let batch_size = x.shape().dims[0];
        let x = x.reshape([batch_size, CHUNK_SIZE * CHARS.len()]).float();
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear3.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear4.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        #[allow(clippy::let_and_return)]
        x
    }
    pub fn forward_regression(&self, x: DataBatch<B>) -> ClassificationOutput<B> {
        let output = self.forward(x.inputs).reshape([-1, CHARS.len() as i32]);
        let targets = x.targets.reshape([-1]);
        let loss = self.loss.forward(output.to_owned(), targets.to_owned());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<DataBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: DataBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_regression(batch);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DataBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: DataBatch<B>) -> ClassificationOutput<B> {
        self.forward_regression(batch)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "CHUNK_SIZE*CHARS.len()")]
    num_classes: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.num_classes, self.num_classes / 2).init(&device),
            linear2: LinearConfig::new(self.num_classes / 2, self.num_classes / 4).init(&device),
            linear3: LinearConfig::new(self.num_classes / 4, self.num_classes / 2).init(&device),
            linear4: LinearConfig::new(self.num_classes / 2, self.num_classes).init(&device),
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            loss: CrossEntropyLossConfig::new().init(&device),
        }
    }
}

#[derive(Clone)]
pub struct DataBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct DataBatch<B: Backend> {
    pub inputs: Tensor<B, 3, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<String, DataBatch<B>> for DataBatcher<B> {
    fn batch(&self, items: Vec<String>) -> DataBatch<B> {
        let vals = items
            .iter()
            .map(|a| {
                let tensors = a
                    .chars()
                    .map(|a| CHARS.find(a).unwrap_or_default())
                    // .map(|c| Tensor::<B, 1>::one_hot(c, CHARS.len(), &self.device))
                    .collect::<Vec<_>>();
                Tensor::<B, 1, Int>::from_ints(&*tensors, &self.device)
            })
            .collect();

        let pre = Tensor::cat(vals, 0).to_device(&self.device);
        let targets = pre.to_owned().reshape([items.len(), CHUNK_SIZE]);
        let inputs = pre
            .one_hot(CHARS.len())
            .reshape([items.len(), CHUNK_SIZE, CHARS.len()]);
        DataBatch { targets, inputs }
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 40)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 1)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

#[derive(Clone)]
pub struct StrDataset(pub String);
impl Dataset<String> for StrDataset {
    fn get(&self, index: usize) -> Option<String> {
        self.0.get(index..(index + CHUNK_SIZE)).map(|a| a.into())
    }

    fn len(&self) -> usize {
        self.0.len() - CHUNK_SIZE
    }
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher1 = DataBatcher::<B>::new(device.clone());
    let batcher2 = DataBatcher::<B::InnerBackend>::new(device.clone());
    let string = include_str!("../../bee.txt");

    let dataloader_train = DataLoaderBuilder::new(batcher1)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(StrDataset(
            string.split_at(string.len() / 5 * 4).0.to_string(),
        ));
    let dataloader_valid = DataLoaderBuilder::new(batcher2)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(StrDataset(
            string.split_at(string.len() / 5 * 4).1.to_string(),
        ));

    let mut lb = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs);
    if let Some(c) = args().nth(1).and_then(|a| a.parse::<usize>().ok()) {
        lb = lb.checkpoint(c)
    }
    let learner = lb.build(
        config.model.init::<B>(device),
        config.optimizer.init(),
        config.learning_rate,
    );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, items: Vec<String>) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(device.clone()).load_record(record);

    let batcher = DataBatcher::new(device);
    let batch = batcher.batch(items.clone());
    let output = model
        .forward(batch.inputs)
        .reshape([items.len(), CHUNK_SIZE, CHARS.len()])
        .argmax(2)
        .into_data();
    let output = output
        .iter::<u16>()
        .map(|a| CHARS.chars().nth(a as usize).unwrap().to_string())
        .collect::<Vec<_>>()
        .join("");

    print!("{}", output);
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    if args().nth(1).is_some_and(|a| a == "infer") {
        let text = if let Some(mut text) = args().nth(2) {
            while text.len() % CHUNK_SIZE != 0 {
                text.push(' ');
            }
            text
        } else {
            include_str!("../../bee.txt").split_at(8192).0.into()
        }
        .as_bytes()
        .chunks(CHUNK_SIZE)
        .map(|a| String::from_utf8_lossy(a).to_string())
        .collect::<Vec<_>>();
        infer::<MyBackend>("/tmp/autoencoder", device, text)
    } else {
        train::<MyAutodiffBackend>(
            "/tmp/autoencoder",
            TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
            device,
        );
    }
}
