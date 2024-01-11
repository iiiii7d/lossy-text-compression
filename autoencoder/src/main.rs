use std::env::args;

use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{Autodiff, Wgpu};
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::nn::loss::{MSELoss, Reduction};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, ReLU};
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;
use burn::train::metric::{CpuUse, LossMetric};
use burn::train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep};

const CHARS: &str = " !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n";
const CHUNK_SIZE: usize = 64;
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    activation: ReLU,
    dropout: Dropout,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
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

        x
    }
    pub fn forward_regression(&self, x: Tensor<B, 2>) -> RegressionOutput<B> {
        let output = self.forward(x.clone());
        let loss = MSELoss::new().forward(output.clone(), x.clone(), Reduction::Auto);

        RegressionOutput::new(loss, output, x)
    }
}

impl<B: AutodiffBackend> TrainStep<Tensor<B, 2>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: Tensor<B, 2>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<Tensor<B, 2>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: Tensor<B, 2>) -> RegressionOutput<B> {
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
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.num_classes, self.num_classes / 2).init(),
            linear2: LinearConfig::new(self.num_classes / 2, self.num_classes / 4).init(),
            linear3: LinearConfig::new(self.num_classes / 4, self.num_classes / 2).init(),
            linear4: LinearConfig::new(self.num_classes / 2, self.num_classes).init(),
            activation: ReLU::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.num_classes, self.num_classes / 2)
                .init_with(record.linear1),
            linear2: LinearConfig::new(self.num_classes / 2, self.num_classes / 4)
                .init_with(record.linear2),
            linear3: LinearConfig::new(self.num_classes / 4, self.num_classes / 2)
                .init_with(record.linear3),
            linear4: LinearConfig::new(self.num_classes / 2, self.num_classes)
                .init_with(record.linear4),
            activation: ReLU::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

pub struct DataBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DataBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<String, Tensor<B, 2>> for DataBatcher<B> {
    fn batch(&self, items: Vec<String>) -> Tensor<B, 2> {
        let vals = items
            .iter()
            .map(|a| {
                let tensors = a
                    .chars()
                    .map(|a| CHARS.find(a).unwrap_or_default())
                    .map(|c| Tensor::<B, 1>::one_hot(c, CHARS.len()))
                    .collect::<Vec<_>>();
                Tensor::cat(tensors, 0).reshape([1, CHARS.len() * CHUNK_SIZE])
            })
            .collect();

        Tensor::cat(vals, 0).to_device(&self.device)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 5)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 0)]
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

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
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
        .load(format!("{artifact_dir}/model").into())
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record).to_device(&device);

    let batcher = DataBatcher::new(device);
    let batch = batcher.batch(items.clone());
    let output = model
        .forward(batch.clone())
        .reshape([items.len(), CHUNK_SIZE, CHARS.len()])
        .argmax(2)
        .into_data()
        .value;
    let output = output
        .into_iter()
        .map(|a| {
            CHARS
                .chars()
                .nth(a.to_string().parse::<usize>().unwrap())
                .unwrap()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join("");

    print!("{}", output);
}

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    if args().nth(1).map_or(false, |a| a == "infer") {
        let text = include_str!("../../shakespeare.txt")
            .split_at(8192)
            .0
            .as_bytes()
            .chunks(CHUNK_SIZE)
            .map(|a| String::from_utf8_lossy(a).to_string())
            .collect::<Vec<_>>();
        infer::<MyBackend>("/tmp/guide", device, text)
    } else {
        train::<MyAutodiffBackend>(
            "/tmp/guide",
            TrainingConfig::new(ModelConfig::new(), AdamConfig::new()),
            device,
        );
    }
}
