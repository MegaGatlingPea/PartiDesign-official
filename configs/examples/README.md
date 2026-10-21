# Standalone examples

The YAML files in this directory are the single source of runnable examples:

- protein.yaml
- peptide.yaml
- nucleic_acids.yaml
- ligand.yaml
- antibody.yaml
- motif.yaml

Their structure inputs live under structures/ and are shared by all examples.
Run from the repository root, for example:

    partidesign design configs/examples/protein.yaml

These files are the maintained public examples location. Benchmark task and
validation exports live under configs/benchmarks/<family>/; the obsolete
The obsolete demo and task copies are no longer part of the public interface.

Fixed full ligand SMILES specify docking, while free or partial chemistry
(scaffold hopping and atom motifs) specifies design. Docking tasks run sampling only
and do not use internal validate/refold gates. Full fixed polymer sequences likewise specify
pose docking. See [validation](../../docs/validation.md#docking-versus-design).

Peptide examples also include TBMB bicyclization and two hydrocarbon staples.
See [peptide chemistry and validation limits](../../docs/tasks/peptide.md#additional-covalent-product-examples)
for the product bonds, source structure, and CPU versus runtime acceptance.
