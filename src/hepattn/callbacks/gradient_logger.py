import math

from lightning import Callback, LightningModule, Trainer


class GradientLoggerCallback(Callback):
    def __init__(self, log_every_n_steps=50, log_parameter_stats=False):
        """Callback to log model gradients during training.

        Args:
            log_every_n_steps (int): Frequency of logging gradients. Logs every `n` steps.
            log_parameter_stats (bool): If True, also logs per-parameter gradient
                norm and std. Disabled by default to reduce logging overhead.
        """
        self.log_every_n_steps = log_every_n_steps
        self.log_parameter_stats = log_parameter_stats
        self._sync_dist = False

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        self._sync_dist = len(trainer.device_ids) > 1

    def on_after_backward(self, trainer, pl_module):
        """Called after the backward pass in training.
        Logs the gradients of the model's parameters.
        """
        if self.log_every_n_steps <= 0:
            return
        # Check if logging should happen at this step
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_sq_norm = 0.0
        total_abs = 0.0
        total_params = 0
        max_abs = 0.0

        for name, param in pl_module.named_parameters():
            grad = param.grad
            if grad is None:
                continue

            grad_detached = grad.detach()
            param_norm = grad_detached.norm(2).item()
            total_sq_norm += param_norm**2

            abs_grad = grad_detached.abs()
            total_abs += abs_grad.sum().item()
            max_abs = max(max_abs, abs_grad.max().item())
            total_params += grad_detached.numel()

            if self.log_parameter_stats:
                pl_module.log(
                    f"grad/{name}/norm",
                    param_norm,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=self._sync_dist,
                )
                pl_module.log(
                    f"grad/{name}/std",
                    grad_detached.std().item(),
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                    sync_dist=self._sync_dist,
                )

        if total_params == 0:
            return

        pl_module.log(
            "grad/global_norm",
            math.sqrt(total_sq_norm),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=self._sync_dist,
        )
        pl_module.log(
            "grad/avg_abs",
            total_abs / total_params,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=self._sync_dist,
        )
        pl_module.log(
            "grad/max_abs",
            max_abs,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=self._sync_dist,
        )
