"""Hydra entrypoint for training the deep kernel acquisition policy."""

import json
import logging
import os

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from meta.config import ExperimentConfig, KernelConfig
from meta.data import TextDataSampler
from meta.policy import AcquisitionPolicy
from meta.surrogate import DKLGaussianProcess, RBFGaussianProcess
from meta.trainer import Trainer

cs = ConfigStore.instance()
cs.store(name="base_config", node=ExperimentConfig)
logger = logging.getLogger(__name__)


def _make_kernel_params(kernel_cfg: KernelConfig) -> dict:
    return {
        "initial_alpha": kernel_cfg.initial_alpha,
        "initial_beta": kernel_cfg.initial_beta,
        "initial_eta": kernel_cfg.initial_eta,
        "learn_priors": kernel_cfg.learn_priors,
    }


def _write_metrics(cfg: ExperimentConfig, metric: float, out_dir: str) -> None:
    payload = {
        "seed": cfg.train.seed,
        "task": cfg.task.name,
        "model": cfg.model.name,
        "acquisition": cfg.acquisition.name,
        "metric": "average_cumulative_gap",
        "value": metric,
    }
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log_path = os.path.join(out_dir, "main.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    generator = torch.Generator().manual_seed(cfg.train.seed)
    sampler = TextDataSampler(cfg.task.path, generator=generator)

    kernel_params = _make_kernel_params(cfg.model.kernel)
    feature_dim = sampler.feature_dim
    if cfg.task.input_dim is not None and cfg.task.input_dim != feature_dim:
        logger.warning(
            "task.input_dim=%s mismatches data feature dim %s; using %s.",
            cfg.task.input_dim,
            feature_dim,
            feature_dim,
        )
    input_dim = feature_dim
    model_name = cfg.model.name.lower()
    if model_name == "dkl_gp":
        surrogate = DKLGaussianProcess(
            input_dim=input_dim,
            feature_dim=cfg.model.network.output_dim,
            hidden_units=cfg.model.network.hidden_units,
            num_layers=cfg.model.network.num_layers,
            kernel_params=kernel_params,
            activation=cfg.model.network.activation,
        )
    elif model_name == "rbf_gp":
        surrogate = RBFGaussianProcess(input_dim=input_dim, kernel_params=kernel_params)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")
    policy = AcquisitionPolicy(surrogate, cfg.acquisition)

    run_dir = HydraConfig.get().runtime.output_dir

    trainer = Trainer(
        sampler,
        policy,
        cfg.task,
        cfg.train,
        pretrain_cfg=cfg.model.pretrain,
        run_dir=run_dir,
        eval_cfg=cfg.evaluation,
    )
    trainer.pretrain()
    trainer.train()

    eval_cfg = cfg.evaluation or getattr(cfg, "evaluation", None)
    if eval_cfg is not None and eval_cfg.test_tasks > 0:
        trainer.maybe_reload_best()
        metric = trainer.evaluate_cumulative_gap(
            eval_cfg.test_tasks, deterministic=True
        )
        _write_metrics(cfg, metric, run_dir)


if __name__ == "__main__":
    main()
