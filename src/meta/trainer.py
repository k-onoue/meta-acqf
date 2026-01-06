"""Meta-training loop for deep kernel acquisition policy."""

import csv
import logging
import os
from typing import List, Optional, Tuple

import torch
from torch import optim
from torch.distributions import Categorical

from meta.config import (EvaluationConfig, PretrainConfig, TaskConfig,
                         TrainConfig)
from meta.data import TextDataSampler
from meta.environment import MetaBOEnvironment
from meta.policy import AcquisitionPolicy

logger = logging.getLogger(__name__)


def _discount_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    return torch.tensor(returns)


class Trainer:
    def __init__(
        self,
        sampler: TextDataSampler,
        policy: AcquisitionPolicy,
        task_cfg: TaskConfig,
        train_cfg: TrainConfig,
        pretrain_cfg: Optional[PretrainConfig] = None,
        run_dir: Optional[str] = None,
        eval_cfg: Optional[EvaluationConfig] = None,
    ) -> None:
        self.sampler = sampler
        self.policy = policy
        self.task_cfg = task_cfg
        self.cfg = train_cfg
        self.pretrain_cfg = pretrain_cfg
        self.eval_cfg = eval_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.generator = torch.Generator().manual_seed(self.cfg.seed)
        self.val_generator = torch.Generator().manual_seed(self.cfg.seed + 999)
        self.run_dir = run_dir or os.getcwd()
        self.loss_csv = os.path.join(self.run_dir, "losses.csv")
        self._init_loss_log()
        self.episode_csv = os.path.join(self.run_dir, "episodes.csv")
        self._init_episode_log()
        self.validation_tasks: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.best_val_gap = float("inf")
        self.best_ckpt_path = os.path.join(self.run_dir, "best_model.pt")
        self._patience_counter = 0
        self._prepare_validation_tasks()

    def pretrain(self) -> None:
        if not self.pretrain_cfg or not self.pretrain_cfg.enabled:
            return

        if not hasattr(self.policy.surrogate, "_nll"):
            logger.warning("Surrogate does not support MLL; skipping pretrain.")
            return

        lr = (
            self.cfg.pretrain_lr
            if self.cfg.pretrain_lr is not None
            else self.pretrain_cfg.lr
        )
        optimizer = optim.Adam(self.policy.surrogate.parameters(), lr=lr)
        best_val = float("inf")
        patience = 0

        for epoch in range(self.pretrain_cfg.epochs):
            X_candidates, y_candidates = self.sampler.sample_task(
                num_candidates=self.task_cfg.candidate_pool_size,
                generator=self.generator,
            )
            train_x = X_candidates.to(self.device)
            train_y = y_candidates.to(self.device)

            optimizer.zero_grad()
            nll = self.policy.surrogate._nll(train_x, train_y)
            nll.backward()
            optimizer.step()

            self._log_loss("pretrain", epoch + 1, float(nll.item()))

            if (epoch + 1) % max(
                1, self.cfg.val_interval
            ) == 0 and self.validation_tasks:
                val_nll = self._evaluate_mll(self.validation_tasks)
                logger.info(
                    "[pretrain] epoch=%s nll=%.4f val_nll=%.4f",
                    epoch + 1,
                    nll.item(),
                    val_nll,
                )
                if val_nll < best_val:
                    best_val = val_nll
                    patience = 0
                    self._save_checkpoint("pretrain_best.pt")
                else:
                    patience += 1
                    if patience >= self.pretrain_cfg.patience:
                        logger.info(
                            "[pretrain] early stopping at epoch=%s (val_nll=%.4f)",
                            epoch + 1,
                            val_nll,
                        )
                        break
            elif (epoch + 1) % max(1, self.pretrain_cfg.epochs // 5) == 0:
                logger.info(
                    "[pretrain] epoch %s/%s nll=%.4f",
                    epoch + 1,
                    self.pretrain_cfg.epochs,
                    nll.item(),
                )

    def train(self) -> None:
        if self.cfg.mode == "greedy":
            self.evaluate_greedy()
            return
        for epoch in range(self.cfg.epochs):
            loss, gap_sum, entropy_mean = self.run_episode()
            loss_value = float(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self._log_loss("train", epoch + 1, loss_value)
            self._log_episode("train", epoch + 1, loss_value, gap_sum, entropy_mean)

            if (epoch + 1) % max(1, self.cfg.epochs // 10) == 0:
                logger.info(
                    "Epoch %s/%s - Loss: %.4f", epoch + 1, self.cfg.epochs, loss_value
                )

            try:
                self._maybe_run_validation(epoch + 1)
            except StopIteration:
                break

    def run_episode(self) -> Tuple[torch.Tensor, float, float]:
        X_candidates, y_candidates = self.sampler.sample_task(
            num_candidates=self.task_cfg.candidate_pool_size
        )
        env = MetaBOEnvironment(
            X_candidates.to(self.device),
            y_candidates.to(self.device),
            initial_pool_size=self.task_cfg.initial_pool_size,
            query_budget=self.task_cfg.query_budget,
            generator=self.generator,
        )

        obs = env.reset()
        xi = torch.tensor(1e-6, device=self.device)
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        entropies: List[torch.Tensor] = []

        done = False
        while not done:
            ctx_x = obs["context_x"].to(self.device)
            ctx_y = obs["context_y"].to(self.device)
            cand_x = obs["candidates_x"].to(self.device)

            if self.cfg.refit_mll and hasattr(self.policy.surrogate, "refit_mll"):
                self.policy.surrogate.refit_mll(
                    ctx_x, ctx_y, steps=self.cfg.mll_steps, lr=self.cfg.mll_lr
                )

            best_y = ctx_y.max()
            scores, probs, xi, var = self.policy(
                ctx_x, ctx_y, cand_x, xi=xi, best_y=best_y
            )

            if self.policy.use_softmax:
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropies.append(dist.entropy())
            else:
                action = scores.argmax()
                log_prob = torch.tensor(0.0, device=self.device)
                entropies.append(torch.tensor(0.0, device=self.device))

            action_idx = int(action.item())
            step_result = env.step(action_idx)
            obs = step_result.observation
            rewards.append(step_result.reward)
            log_probs.append(log_prob)
            done = step_result.done

            # Update xi for MI using variance of chosen action
            xi = xi + var[action_idx]

        returns = _discount_returns(rewards, gamma=self.policy.acq_config.gamma).to(
            self.device
        )
        if len(returns) > 1:
            advantage = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            advantage = returns - returns.mean()

        log_probs_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)

        pg_loss = -(log_probs_tensor * advantage.detach()).sum()

        ent_coef = getattr(self.cfg, "ent_coef", 0.01)
        entropy_loss = -ent_coef * entropy_tensor.sum()

        loss = pg_loss + entropy_loss

        gap_sum = -float(sum(rewards))
        entropy_mean = float(torch.stack(entropies).mean().item()) if entropies else 0.0

        return loss, gap_sum, entropy_mean

    def evaluate_greedy(self) -> None:
        rewards_all: List[float] = []
        episodes = max(1, self.cfg.epochs)
        for _ in range(episodes):
            X_candidates, y_candidates = self.sampler.sample_task(
                num_candidates=self.task_cfg.candidate_pool_size
            )
            env = MetaBOEnvironment(
                X_candidates.to(self.device),
                y_candidates.to(self.device),
                initial_pool_size=self.task_cfg.initial_pool_size,
                query_budget=self.task_cfg.query_budget,
                generator=self.generator,
            )

            obs = env.reset()
            xi = torch.tensor(1e-6, device=self.device)
            cumulative_reward = 0.0
            done = False
            with torch.set_grad_enabled(self.cfg.refit_mll):
                while not done:
                    ctx_x = obs["context_x"].to(self.device)
                    ctx_y = obs["context_y"].to(self.device)
                    cand_x = obs["candidates_x"].to(self.device)

                    if self.cfg.refit_mll and hasattr(
                        self.policy.surrogate, "refit_mll"
                    ):
                        self.policy.surrogate.refit_mll(
                            ctx_x, ctx_y, steps=self.cfg.mll_steps, lr=self.cfg.mll_lr
                        )

                    best_y = ctx_y.max()
                    scores, _, xi, var = self.policy(
                        ctx_x, ctx_y, cand_x, xi=xi, best_y=best_y
                    )

                    action_idx = int(scores.argmax().item())
                    step_result = env.step(action_idx)
                    cumulative_reward += step_result.reward
                    obs = step_result.observation
                    xi = xi + var[action_idx]
                    done = step_result.done

            rewards_all.append(cumulative_reward)

            self._log_episode("greedy", len(rewards_all), 0.0, -cumulative_reward, 0.0)

        avg_reward = sum(rewards_all) / len(rewards_all)
        logger.info("[greedy] episodes=%s avg_return=%.4f", episodes, avg_reward)

    def evaluate_cumulative_gap(
        self,
        num_tasks: int,
        deterministic: bool = True,
        tasks: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        allow_refit: bool = True,
    ) -> float:
        gaps: List[float] = []
        per_step_gaps: List[List[float]] = []
        per_task_records: List[tuple[int, int, float]] = []
        task_list = tasks if tasks is not None else [None] * num_tasks
        if tasks is None:
            task_list = [None for _ in range(num_tasks)]
        else:
            num_tasks = len(tasks)

        for task_id, preset in enumerate(task_list):
            if preset is None:
                X_candidates, y_candidates = self.sampler.sample_task(
                    num_candidates=self.task_cfg.candidate_pool_size,
                    generator=self.generator,
                )
            else:
                X_candidates, y_candidates = preset
            env = MetaBOEnvironment(
                X_candidates.to(self.device),
                y_candidates.to(self.device),
                initial_pool_size=self.task_cfg.initial_pool_size,
                query_budget=self.task_cfg.query_budget,
                generator=self.generator,
            )

            obs = env.reset()
            xi = torch.tensor(1e-6, device=self.device)
            best_so_far = -float("inf")
            history_best: List[float] = []
            true_opt = float(y_candidates.max().item())
            done = False
            with torch.set_grad_enabled(self.cfg.refit_mll and allow_refit):
                while not done:
                    ctx_x = obs["context_x"].to(self.device)
                    ctx_y = obs["context_y"].to(self.device)
                    cand_x = obs["candidates_x"].to(self.device)

                    if (
                        allow_refit
                        and self.cfg.refit_mll
                        and hasattr(self.policy.surrogate, "refit_mll")
                    ):
                        self.policy.surrogate.refit_mll(
                            ctx_x, ctx_y, steps=self.cfg.mll_steps, lr=self.cfg.mll_lr
                        )

                    best_y = ctx_y.max()
                    scores, probs, xi, var = self.policy(
                        ctx_x, ctx_y, cand_x, xi=xi, best_y=best_y
                    )

                    if deterministic or not self.policy.use_softmax:
                        action_idx = int(scores.argmax().item())
                    else:
                        dist = Categorical(probs)
                        action_idx = int(dist.sample().item())

                    step_result = env.step(action_idx)
                    obs = step_result.observation
                    xi = xi + var[action_idx]
                    best_so_far = max(
                        best_so_far,
                        float(step_result.observation["context_y"].max().item()),
                    )
                    history_best.append(best_so_far)
                    done = step_result.done

            gaps.append(self._average_cumulative_gap(history_best, true_opt))
            # Track per-step gaps for aggregation
            task_gaps = [true_opt - b for b in history_best]
            for step_idx, gap in enumerate(task_gaps, start=1):
                per_task_records.append((task_id, step_idx, gap))
            if len(per_step_gaps) < len(task_gaps):
                per_step_gaps.extend(
                    [] for _ in range(len(task_gaps) - len(per_step_gaps))
                )
            for i, g in enumerate(task_gaps):
                per_step_gaps[i].append(g)

        avg_gap = float(sum(gaps) / len(gaps))
        self._write_gap_trace(per_step_gaps)
        self._write_gap_per_task_trace(per_task_records)
        return avg_gap

    @staticmethod
    def _average_cumulative_gap(history_best: List[float], true_opt: float) -> float:
        if not history_best:
            return 0.0
        gaps = [true_opt - b for b in history_best]
        return float(sum(gaps) / len(history_best))

    def _init_loss_log(self) -> None:
        if os.path.exists(self.loss_csv):
            return
        with open(self.loss_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["phase", "step", "loss"])

    def _log_loss(self, phase: str, step: int, loss: float) -> None:
        with open(self.loss_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([phase, step, f"{loss:.6f}"])

    def _init_episode_log(self) -> None:
        if os.path.exists(self.episode_csv):
            return
        with open(self.episode_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["phase", "episode", "loss", "gap_sum", "entropy"])

    def _log_episode(
        self, phase: str, episode: int, loss: float, gap_sum: float, entropy: float
    ) -> None:
        with open(self.episode_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [phase, episode, f"{loss:.6f}", f"{gap_sum:.6f}", f"{entropy:.6f}"]
            )

    def _write_gap_trace(self, per_step_gaps: List[List[float]]) -> None:
        if not per_step_gaps:
            return
        out_path = os.path.join(self.run_dir, "gaps.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "avg_gap"])
            for step_idx, vals in enumerate(per_step_gaps, start=1):
                avg_gap = sum(vals) / len(vals) if vals else 0.0
                writer.writerow([step_idx, f"{avg_gap:.6f}"])

    def _write_gap_per_task_trace(self, records: List[tuple[int, int, float]]) -> None:
        if not records:
            return
        out_path = os.path.join(self.run_dir, "gaps_per_task.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "step", "gap"])
            for task_id, step, gap in records:
                writer.writerow([task_id, step, f"{gap:.6f}"])

    def _evaluate_mll(self, tasks: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        if not hasattr(self.policy.surrogate, "_nll"):
            return float("inf")
        nlls: List[float] = []
        self.policy.eval()
        try:
            for X, y in tasks:
                with torch.no_grad():
                    nll = self.policy.surrogate._nll(
                        X.to(self.device), y.to(self.device)
                    )
                nlls.append(float(nll.item()))
        finally:
            self.policy.train()
        return float(sum(nlls) / len(nlls)) if nlls else float("inf")

    def _prepare_validation_tasks(self) -> None:
        if not self.eval_cfg or self.eval_cfg.validation_tasks <= 0:
            return
        self.validation_tasks = []
        for _ in range(self.eval_cfg.validation_tasks):
            X_candidates, y_candidates = self.sampler.sample_task(
                num_candidates=self.task_cfg.candidate_pool_size,
                generator=self.val_generator,
            )
            # Store CPU copies to avoid accidental mutation on device
            self.validation_tasks.append((X_candidates.clone(), y_candidates.clone()))

    def _maybe_run_validation(self, epoch: int) -> None:
        if not self.eval_cfg or self.eval_cfg.validation_tasks <= 0:
            return
        if epoch % max(1, self.cfg.val_interval) != 0:
            return

        self.policy.eval()
        try:
            val_gap = self.evaluate_cumulative_gap(
                num_tasks=self.eval_cfg.validation_tasks,
                deterministic=True,
                tasks=self.validation_tasks,
                allow_refit=False,
            )
            logger.info("[val] epoch=%s gap=%.4f", epoch, val_gap)

            improved = val_gap < self.best_val_gap
            if improved:
                self.best_val_gap = val_gap
                self._patience_counter = 0
                self._save_checkpoint("best_model.pt")
            else:
                self._patience_counter += 1
                if (
                    self.cfg.early_stopping_patience is not None
                    and self._patience_counter >= self.cfg.early_stopping_patience
                ):
                    logger.info(
                        "[val] early stopping at epoch=%s (gap=%.4f)", epoch, val_gap
                    )
                    raise StopIteration
        finally:
            self.policy.train()

    def _save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.run_dir, filename)
        torch.save(self.policy.state_dict(), path)

    def maybe_reload_best(self) -> None:
        if not getattr(self.cfg, "reload_best", True):
            return
        if not os.path.exists(self.best_ckpt_path):
            logger.warning("best_model.pt not found; skipping reload")
            return
        state = torch.load(self.best_ckpt_path, map_location=self.device)
        self.policy.load_state_dict(state)
        logger.info("Reloaded best_model.pt for evaluation")
