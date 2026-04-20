import math
import re
from collections import defaultdict

from lightning import Callback, LightningModule, Trainer


# Parameter name patterns used to group grads into logical submodules.
# Order matters — the first matching pattern wins. The regexes are compiled
# once at import time. The resulting label is what appears in the metric key
# ``grad/<label>/{norm,avg_abs,max_abs}``.
#
# Encoder layers are matched dynamically so the logger works for any depth —
# the ``encoder_layer_<i>`` label carries the layer index. Everything outside
# these patterns is aggregated under ``other``.
_LAYER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:model\.)?input_net\."),    "input_net"),
    (re.compile(r"^(?:model\.)?output_head\."),  "output_head"),
    (re.compile(r"^(?:model\.)?pool_head\."),    "pool_head"),
    (re.compile(r"^(?:model\.)?fwd_head\."),     "fwd_head"),
    (re.compile(r"^(?:model\.)?bwd_head\."),     "bwd_head"),
]
_ENCODER_LAYER_RE = re.compile(r"^(?:model\.)?encoder\.layers\.(\d+)\.")
_ENCODER_OTHER_RE = re.compile(r"^(?:model\.)?encoder\.")


def _group_for(name: str) -> str:
    """Map a parameter name to a submodule group label."""
    for pat, label in _LAYER_PATTERNS:
        if pat.match(name):
            return label
    m = _ENCODER_LAYER_RE.match(name)
    if m is not None:
        return f"encoder_layer_{int(m.group(1)):02d}"
    if _ENCODER_OTHER_RE.match(name):
        return "encoder_other"
    return "other"


class GradientLoggerCallback(Callback):
    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_parameter_stats: bool = False,
        log_layer_stats: bool = True,
    ):
        """Log gradient statistics during training.

        Always logs the three global metrics (`grad/global_norm`,
        `grad/avg_abs`, `grad/max_abs`). When ``log_layer_stats`` is set
        (default), the same three metrics are also logged under
        ``grad/<group>/...`` for each model submodule group:

        - ``input_net``, ``output_head``, ``pool_head``, ``fwd_head``,
          ``bwd_head`` — the Dense heads around the encoder.
        - ``encoder_layer_00``, ``encoder_layer_01``, ... — one group per
          encoder layer (matched via ``encoder.layers.<i>.*``, works for any
          depth).
        - ``encoder_other`` — any encoder-level parameter outside
          ``encoder.layers`` (e.g. trunk-level norms, CLS tokens).
        - ``other`` — anything that did not match the above.

        Args:
            log_every_n_steps: frequency of logging. ``0`` disables entirely.
            log_parameter_stats: also log norm+std per individual parameter
                tensor. Off by default — produces a very large number of
                series.
            log_layer_stats: aggregate grads per submodule group as described
                above. On by default so deeper networks come with per-layer
                visibility out of the box.
        """
        self.log_every_n_steps = log_every_n_steps
        self.log_parameter_stats = log_parameter_stats
        self.log_layer_stats = log_layer_stats
        self._sync_dist = False

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        self._sync_dist = len(trainer.device_ids) > 1

    def on_after_backward(self, trainer, pl_module):
        if self.log_every_n_steps <= 0:
            return
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total_sq_norm = 0.0
        total_abs = 0.0
        total_params = 0
        max_abs = 0.0

        # Per-group accumulators.
        group_sq_norm: dict[str, float] = defaultdict(float)
        group_abs:     dict[str, float] = defaultdict(float)
        group_params:  dict[str, int]   = defaultdict(int)
        group_max:     dict[str, float] = defaultdict(float)

        for name, param in pl_module.named_parameters():
            grad = param.grad
            if grad is None:
                continue

            grad_detached = grad.detach()
            param_norm = grad_detached.norm(2).item()
            abs_grad = grad_detached.abs()
            abs_sum = abs_grad.sum().item()
            abs_max = abs_grad.max().item()
            numel = grad_detached.numel()

            total_sq_norm += param_norm ** 2
            total_abs += abs_sum
            total_params += numel
            max_abs = max(max_abs, abs_max)

            if self.log_layer_stats:
                g = _group_for(name)
                group_sq_norm[g] += param_norm ** 2
                group_abs[g]     += abs_sum
                group_params[g]  += numel
                if abs_max > group_max[g]:
                    group_max[g] = abs_max

            if self.log_parameter_stats:
                pl_module.log(
                    f"grad/{name}/norm", param_norm,
                    on_step=True, on_epoch=False, logger=True,
                    sync_dist=self._sync_dist,
                )
                pl_module.log(
                    f"grad/{name}/std", grad_detached.std().item(),
                    on_step=True, on_epoch=False, logger=True,
                    sync_dist=self._sync_dist,
                )

        if total_params == 0:
            return

        pl_module.log(
            "grad/global_norm", math.sqrt(total_sq_norm),
            on_step=True, on_epoch=False, logger=True,
            sync_dist=self._sync_dist,
        )
        pl_module.log(
            "grad/avg_abs", total_abs / total_params,
            on_step=True, on_epoch=False, logger=True,
            sync_dist=self._sync_dist,
        )
        pl_module.log(
            "grad/max_abs", max_abs,
            on_step=True, on_epoch=False, logger=True,
            sync_dist=self._sync_dist,
        )

        if not self.log_layer_stats:
            return

        for g, n in group_params.items():
            if n == 0:
                continue
            pl_module.log(
                f"grad/{g}/norm", math.sqrt(group_sq_norm[g]),
                on_step=True, on_epoch=False, logger=True,
                sync_dist=self._sync_dist,
            )
            pl_module.log(
                f"grad/{g}/avg_abs", group_abs[g] / n,
                on_step=True, on_epoch=False, logger=True,
                sync_dist=self._sync_dist,
            )
            pl_module.log(
                f"grad/{g}/max_abs", group_max[g],
                on_step=True, on_epoch=False, logger=True,
                sync_dist=self._sync_dist,
            )
