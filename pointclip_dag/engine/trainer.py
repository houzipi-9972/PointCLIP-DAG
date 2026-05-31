from __future__ import annotations

import csv
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from pointclip_dag.engine.evaluator import Evaluator
from pointclip_dag.utils.checkpoint import save_checkpoint
from pointclip_dag.utils.misc import move_to_device


class Trainer:
    def __init__(
        self,
        cfg,
        model,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        train_vocabulary,
        eval_vocabulary,
        device,
        run_dir,
        logger,
        start_step=0,
        best_miou=-1.0,
    ):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_vocabulary = train_vocabulary
        self.eval_vocabulary = eval_vocabulary
        self.device = device
        self.run_dir = Path(run_dir)
        self.logger = logger
        self.global_step = int(start_step)
        self.best_miou = float(best_miou)
        self.log_dir = self.run_dir / "logs"
        self.loss_csv = self.log_dir / "loss_history.csv"
        self.loss_png = self.log_dir / "loss_curves.png"
        self.curve_dir = self.log_dir / "loss_curves"
        self._plot_warning_logged = False
        self._init_loss_history()

    def train(self):
        self.model.to(self.device)
        max_epochs = int(self.cfg.train.epochs)
        max_iters = int(self.cfg.train.get("max_iters", 0))
        self.logger.info(
            "start training: epochs=%d max_iters=%d train_batches=%d device=%s",
            max_epochs,
            max_iters,
            len(self.train_loader),
            self.device,
        )
        scheduler_unit = self.cfg.get("scheduler", {}).get("step_unit", "iter")
        for epoch in range(max_epochs):
            self.model.train()
            self.model.set_vocabulary(self.train_vocabulary)
            self.logger.info("epoch=%d fetching batches...", epoch)
            total = len(self.train_loader)
            if max_iters:
                total = min(total, max(0, max_iters - self.global_step))
            progress = tqdm(
                self.train_loader,
                total=total,
                desc=f"epoch {epoch + 1}/{max_epochs}",
                dynamic_ncols=True,
                disable=not bool(self.cfg.train.get("progress_bar", True)),
            )
            for batch in progress:
                if max_iters and self.global_step >= max_iters:
                    break
                self.global_step += 1
                if self.global_step == 1:
                    self.logger.info(
                        "first batch loaded: points=%s image=%s sparse_depth=%s datasets=%s",
                        tuple(batch["points"].shape),
                        tuple(batch["image"].shape),
                        tuple(batch["sparse_depth"].shape),
                        batch.get("dataset_name", []),
                    )
                batch = move_to_device(batch, self.device)
                if self.global_step == 1:
                    self.logger.info("first batch moved to %s; running forward/backward...", self.device)
                outputs = self.model(batch)
                losses = self.loss_fn(outputs, batch)
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if self._should_debug():
                    self._log_debug_before_step(outputs, losses)
                if self.cfg.train.get("grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.train.grad_clip_norm))
                if self._should_debug() and self.cfg.get("debug", {}).get("check_grad_norm", True):
                    self._log_grad_norms()
                self.optimizer.step()
                if self.scheduler is not None and scheduler_unit == "iter":
                    self.scheduler.step()
                if self.global_step % int(self.cfg.train.log_period) == 0:
                    values = self._loss_values(losses)
                    lr = self._current_lr()
                    self._log_losses(epoch, values, lr)
                    self._append_loss_history(epoch, values, lr)
                    self._maybe_write_loss_curves()
                    progress.set_postfix(self._progress_values(values, lr))
                if self.global_step % int(self.cfg.train.checkpoint_period) == 0:
                    self._save("latest.pth", epoch)
                if self.val_loader is not None and self.global_step % int(self.cfg.eval.period) == 0:
                    self._validate(epoch)
                if max_iters and self.global_step >= max_iters:
                    self._maybe_write_loss_curves(force=True)
                    self._save("final.pth", epoch)
                    return
            if self.scheduler is not None and scheduler_unit == "epoch":
                self.scheduler.step()
        self._maybe_write_loss_curves(force=True)
        self._save("final.pth", max_epochs - 1)

    def _init_loss_history(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.global_step > 0 and self.loss_csv.exists():
            return
        with self.loss_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step",
                    "epoch",
                    "lr",
                    "loss",
                    "loss_3d_ce",
                    "loss_2d_ce",
                    "loss_2d_coarse_ce",
                    "loss_feat",
                    "loss_kl",
                    "metric_2d_projected_acc",
                    "metric_2d_projected_miou",
                    "metric_valid_projected_ratio",
                    "metric_ignored_projected_label_ratio",
                    "metric_supervised_projected_ratio",
                    "metric_distill_projected_ratio",
                ]
            )

    def _loss_values(self, losses):
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    def _log_losses(self, epoch, values, lr):
        self.logger.info("epoch=%d step=%d lr=%.6g %s", epoch, self.global_step, lr, values)

    def _append_loss_history(self, epoch, values, lr):
        with self.loss_csv.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    self.global_step,
                    epoch,
                    lr,
                    values.get("loss", np.nan),
                    values.get("loss_3d_ce", np.nan),
                    values.get("loss_2d_ce", np.nan),
                    values.get("loss_2d_coarse_ce", np.nan),
                    values.get("loss_feat", np.nan),
                    values.get("loss_kl", np.nan),
                    values.get("metric_2d_projected_acc", np.nan),
                    values.get("metric_2d_projected_miou", np.nan),
                    values.get("metric_valid_projected_ratio", np.nan),
                    values.get("metric_ignored_projected_label_ratio", np.nan),
                    values.get("metric_supervised_projected_ratio", np.nan),
                    values.get("metric_distill_projected_ratio", np.nan),
                ]
            )

    def _maybe_write_loss_curves(self, force=False):
        curve_period = int(self.cfg.train.get("curve_period", 500))
        if not force and (curve_period <= 0 or self.global_step % curve_period != 0):
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            if not self._plot_warning_logged:
                self.logger.warning("loss curve png disabled: matplotlib import failed: %s", exc)
                self._plot_warning_logged = True
            return

        data = np.genfromtxt(self.loss_csv, delimiter=",", names=True)
        if data.size == 0:
            return
        data = np.atleast_1d(data)
        steps = data["step"]
        loss_names = ["loss", "loss_3d_ce", "loss_2d_ce", "loss_2d_coarse_ce", "loss_feat", "loss_kl"]
        self.curve_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        for name in loss_names:
            values = data[name]
            mask = np.isfinite(values)
            if np.any(mask):
                plt.plot(steps[mask], values[mask], label=name)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.loss_png, dpi=160)
        plt.close()
        for name in loss_names:
            values = data[name]
            mask = np.isfinite(values)
            if not np.any(mask):
                continue
            plt.figure(figsize=(8, 5))
            plt.plot(steps[mask], values[mask], label=name)
            plt.xlabel("step")
            plt.ylabel(name)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.curve_dir / f"{name}.png", dpi=160)
            plt.close()

    def _progress_values(self, values, lr):
        return {
            "loss": f"{values.get('loss', 0.0):.3f}",
            "3d": f"{values.get('loss_3d_ce', 0.0):.3f}",
            "2d": f"{values.get('loss_2d_ce', 0.0):.3f}",
            "2d_coarse": f"{values.get('loss_2d_coarse_ce', 0.0):.3f}",
            "2d_acc": f"{values.get('metric_2d_projected_acc', 0.0):.3f}",
            "feat": f"{values.get('loss_feat', 0.0):.3f}",
            "kl": f"{values.get('loss_kl', 0.0):.3f}",
            "lr": f"{lr:.2e}",
        }

    def _current_lr(self):
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _should_debug(self):
        debug = self.cfg.get("debug", {})
        if not debug.get("enabled", False):
            return False
        first_n = int(debug.get("log_first_n_steps", 5))
        interval = int(debug.get("log_interval", 500))
        return self.global_step <= first_n or (interval > 0 and self.global_step % interval == 0)

    def _log_debug_before_step(self, outputs, losses):
        labels = outputs["labels_vocab"].detach()
        raw = outputs["raw_point_labels"].detach()
        ignore = int(self.cfg.loss.ignore_index)
        valid = labels != ignore
        ignore_ratio = 1.0 - float(valid.float().mean().item()) if labels.numel() else 1.0
        logits = outputs["logits3d"].detach()
        z3d_norm = outputs["z3d"].detach().norm(dim=-1)
        text_norm = outputs["text_embeddings"].detach().norm(dim=-1)
        pred = logits.argmax(dim=-1)
        self.logger.info(
            "debug step=%d train_vocab=%d names=%s",
            self.global_step,
            self.train_vocabulary.num_classes,
            self.train_vocabulary.names,
        )
        self.logger.info(
            "debug labels raw_unique=%s mapped_unique=%s ignore_ratio=%.4f mapped_hist=%s pred_hist=%s",
            _unique_cpu(raw),
            _unique_cpu(labels),
            ignore_ratio,
            _hist_cpu(labels[valid], self.train_vocabulary.num_classes),
            _hist_cpu(pred, self.train_vocabulary.num_classes),
        )
        self.logger.info(
            "debug logits shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f z3d_norm=%.4f/%.4f text_norm=%.4f/%.4f losses=%s",
            tuple(logits.shape),
            float(logits.min().item()),
            float(logits.max().item()),
            float(logits.mean().item()),
            float(logits.std().item()),
            float(z3d_norm.mean().item()),
            float(z3d_norm.std().item()),
            float(text_norm.mean().item()),
            float(text_norm.std().item()),
            {key: float(value.detach().cpu()) for key, value in losses.items()},
        )
        if "logits2d_points" in outputs and outputs["logits2d_points"].numel() > 0:
            labels_valid = outputs["labels_valid"].detach()
            valid_2d = labels_valid != ignore
            pred2d = outputs["logits2d_points"].detach().argmax(dim=-1)
            self.logger.info(
                "debug 2d valid_projected_ratio=%.4f ignored_projected_label_ratio=%.4f pred_hist=%s gt_hist=%s depth_mode=%s",
                float(losses.get("metric_valid_projected_ratio", logits.sum() * 0.0).detach().cpu()),
                float(losses.get("metric_ignored_projected_label_ratio", logits.sum() * 0.0).detach().cpu()),
                _hist_cpu(pred2d[valid_2d], self.train_vocabulary.num_classes),
                _hist_cpu(labels_valid[valid_2d], self.train_vocabulary.num_classes),
                outputs.get("branch2d_depth_mode", ""),
            )

    def _log_grad_norms(self):
        self.logger.info(
            "debug grad_norm branch3d=%.4f projection_head=%.4f branch2d=%.4f",
            _grad_norm(self.model.branch3d),
            _grad_norm(self.model.head3d),
            _grad_norm(self.model.branch2d),
        )

    def _validate(self, epoch):
        evaluator = Evaluator(
            self.cfg,
            self.model,
            self.val_loader,
            self.eval_vocabulary,
            self.device,
            logger=self.logger,
            out_dir=self.run_dir / "eval",
            train_vocabulary=self.train_vocabulary,
        )
        metrics = evaluator.evaluate(save_predictions=False)
        miou = metrics["all_mIoU"]
        if miou > self.best_miou:
            self.best_miou = miou
            self._save("best.pth", epoch, metrics=metrics)
        self.model.train()
        self.model.set_vocabulary(self.train_vocabulary)

    def _save(self, name, epoch, **extra):
        save_checkpoint(
            self.run_dir / "checkpoints" / name,
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=epoch,
            global_step=self.global_step,
            best_miou=self.best_miou,
            **extra,
        )


def _unique_cpu(tensor):
    return torch.unique(tensor.detach().cpu()).tolist()


def _hist_cpu(tensor, num_classes):
    if tensor.numel() == 0:
        return []
    values = tensor.detach().cpu().numpy().astype(np.int64)
    values = values[(values >= 0) & (values < num_classes)]
    return np.bincount(values, minlength=num_classes).tolist()


def _grad_norm(module):
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().norm(2).item() ** 2)
    return total ** 0.5
