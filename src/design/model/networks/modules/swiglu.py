import torch
from torch.nn import functional as F


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class SwiGLU(torch.nn.Module):
    """SwiGLU layer."""

    def __init__(self):
        super().__init__()
        self._bioir_backend = None

    def forward(self, x):
        """
        Args:
            x: input tensor, shape [..., d]

        Returns:
            Tensor of shape [..., d//2].
        """
        if self._bioir_backend is not None and not self.training and not torch.is_grad_enabled():
            return self._bioir_backend.swiglu(x)
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x
