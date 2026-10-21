# AF2 fast triangle-multiplication adapter

The copied Pallas sources are from
`https://github.com/anthropics/uplifting-biomolecular-modeling` at revision
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`:

- `trimul_pallas.py` from `common/opt_core/opt_core/kernels/fpf_pallas/trimul_pallas.py`;
- `fpf_pallas_bias.py` from `common/opt_core/opt_core/kernels/fpf_pallas_bias.py`;
- `fpf_pallas.NOTICE` and `opt_core.LICENSE` carry the source attribution and
  Apache-2.0 license text.

`fpf_pallas_bias.py` is adapted only at its triangle prologue/epilogue block
specifications so AF2's input channel width and intermediate width may differ
(the shipped AF2 configurations use 128 and 64).  The LayerNorm, sigmoid,
mask, bias, contraction and output-gate arithmetic remains the pinned source
operation order.  `fpf_pallas_serve.py` is a minimal local carrier shim; no
attention provider or optimization-kit framework is copied.

`api.py` preserves the vendor module's contiguous `[left | right]` parameter
layout by converting it to the copied kernel's interleaved layout.  It pads
only the residue axes to the kernel tile alignment and crops the result.  CPU
and non-bf16 calls use the native statement, while GPU bf16 calls use Pallas.
