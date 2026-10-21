# Runtime resources

Static inference data lives here. Runtime code resolves this directory from the
checkout or the installed package, independently of the working directory and
selected model-weight directory.

- `chemistry/ccd.json.gz`: the single native chemical-component table.
- `ligand/ligand_atom_count_distribution.json`: pocket-dependent ligand size sampling.
- `model/ae_config.yaml`: the fixed autoencoder architecture configuration.
- `antibody/scaffolds/`: the 14 Fab and 7 VHH YAML/CIF pairs; their license is in `antibody/LICENSE`.
- `openmm/uff_nonbonded.tsv`: the pinned UFF parameter subset; its license is adjacent.

Keep each antibody YAML beside its referenced CIF. Model checkpoints remain in
`ckpts/` and `community_models/`; they are not resources in this directory.
