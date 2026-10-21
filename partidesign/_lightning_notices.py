"""Filters for two optional Lightning CLI startup notices."""
from __future__ import annotations

import logging
import re
import warnings


_TENSOR_CORE_NOTICE = re.compile(
    r"You are using a CUDA device \([^\n]+\) "
    + re.escape(
        "that has Tensor Cores. To properly utilize them, you should set "
        "`torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off "
        "precision for performance. For more details, read "
        "https://pytorch.org/docs/stable/generated/"
        "torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"
    )
)
_SRUN_NOTICE = (
    r"The `srun` command is available on your system but is not used\. HINT: "
    r"If your intention is to run Lightning on SLURM, prepend your python command "
    r"with `srun` like so: .+ \.\.\.\Z"
)
_RANK_ZERO_LOGGERS = (
    "lightning.fabric.utilities.rank_zero",
    "lightning.pytorch.utilities.rank_zero",
)


class _TensorCoreNoticeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.levelno == logging.INFO
            and _TENSOR_CORE_NOTICE.fullmatch(record.getMessage()) is not None
        )


_TENSOR_CORE_FILTER = _TensorCoreNoticeFilter()


def configure_lightning_notices() -> None:
    """Hide the exact Tensor Core INFO and unused-srun UserWarning notices."""
    for name in _RANK_ZERO_LOGGERS:
        logging.getLogger(name).addFilter(_TENSOR_CORE_FILTER)
    warnings.filterwarnings(
        "ignore",
        message=_SRUN_NOTICE,
        category=UserWarning,
        module=r"lightning\.fabric\.plugins\.environments\.slurm\Z",
    )
