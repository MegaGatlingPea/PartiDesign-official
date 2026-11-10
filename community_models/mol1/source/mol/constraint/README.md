# Mol-1 Constraint

`mol-1-constraint` adds SITE, CONTACT and EXCLUSION conditioning to Mol-1.
The released model supports all three; each input may provide any combination
or none. Conditioning guides predictions without guaranteeing satisfaction.

- [Prediction and weight download](https://github.com/molence/Mol-1/blob/9e85ad16b3d7031dd92c34d99f895ee65547038b/README.md#predict-with-mol-1-constraint)
- [Constraint semantics and input example](https://github.com/molence/Mol-1/blob/9e85ad16b3d7031dd92c34d99f895ee65547038b/docs/constraint_semantics.md)
- [Evidence-to-constraints skill](https://github.com/molence/Mol-1/blob/9e85ad16b3d7031dd92c34d99f895ee65547038b/docs/mol1-evidence-to-constraints/SKILL.md)
- [Prepared 7PZB example](https://github.com/molence/Mol-1/blob/9e85ad16b3d7031dd92c34d99f895ee65547038b/examples/7pzb_constraint_binding_site/input.json)

[`config.py`](config.py) holds inference defaults.
No configuration file is required; `--config` JSON overrides remain supported.
