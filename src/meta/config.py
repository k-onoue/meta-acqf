"""Configuration dataclasses used with Hydra for meta-BO experiments."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskConfig:
    name: str
    path: str
    input_dim: Optional[int] = None
    num_classes: Optional[int] = None
    initial_pool_size: int = 20
    query_budget: int = 10
    candidate_pool_size: int = 500


@dataclass
class NetworkConfig:
    hidden_units: int = 32
    num_layers: int = 4
    output_dim: int = 32
    activation: str = "relu"


@dataclass
class KernelConfig:
    initial_alpha: float = 1.0
    initial_beta: float = 0.01
    initial_eta: float = 1.0
    learn_priors: bool = True


@dataclass
class PretrainConfig:
    enabled: bool = False
    epochs: int = 100
    lr: float = 1e-2
    patience: int = 20


@dataclass
class EvaluationConfig:
    metric: str = "average_cumulative_gap"
    seeds: int = 10
    test_tasks: int = 50
    validation_tasks: int = 20


@dataclass
class ModelConfig:
    name: str
    network: NetworkConfig
    kernel: KernelConfig
    pretrain: Optional[PretrainConfig] = None


@dataclass
class AcquisitionConfig:
    name: str  # "MI", "EI", "UCB"
    gamma: float = 0.99
    nu: float = 0.069
    use_softmax: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 1000
    optimizer: str = "Adam"
    seed: int = 42
    gpu_id: int = 0
    max_tasks_per_step: int = 1
    mode: str = "rl"  # "rl" or "greedy"
    refit_mll: bool = False
    mll_steps: int = 150
    mll_lr: float = 1e-2
    ent_coef: float = 0.01
    val_interval: int = 10
    early_stopping_patience: Optional[int] = 20
    pretrain_lr: Optional[float] = None
    reload_best: bool = True


@dataclass
class ExperimentConfig:
    task: TaskConfig
    model: ModelConfig
    acquisition: AcquisitionConfig
    train: TrainConfig
    debug: Optional[bool] = False
    evaluation: Optional[EvaluationConfig] = None
