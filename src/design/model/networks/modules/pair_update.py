"""Guard for pair-update modes absent from the released checkpoint ABI.

The combined FM checkpoint and its fixed VAE construction configuration both use
``update_pair_repr=false`` and ``use_tri_mult=false``.  Keeping this explicit guard
avoids importing the training-only OpenFold pair stack while still producing a useful
error if an incompatible checkpoint configuration is supplied.
"""

import torch


class PairReprUpdate(torch.nn.Module):
    """Reject checkpoint configurations outside the fixed standalone ABI."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        super().__init__()
        raise ValueError(
            "standalone inference requires update_pair_repr=false and "
            "use_tri_mult=false"
        )
