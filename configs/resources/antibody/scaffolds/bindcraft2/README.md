# VHH scaffold data

`VHH.cif` is imported from `scaffolds/VHH.cif` in
[PacesaLab/BindCraft2](https://github.com/PacesaLab/BindCraft2/tree/b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06),
revision `b6fb3ae15ab6a702f1156cffc11bcc3323c2fd06`. The upstream copyright and
[BindCraft2 Source-Available License (Hosting-Restricted)](LICENSE) are retained
verbatim. These imported data are not represented as MIT-licensed. No upstream
runtime is included.

Modification notice: the source CIF lacks `entity_poly` and `entity_poly_seq`.
These categories were added once from its complete observed 127-residue standard
protein sequence. Every original atom-site column, coordinate, identity,
occupancy, model, author identifier and insertion code is retained. The source
uses label chain `B`, author chain `A`, and label/author residues 1–127.

`bindcraft2_vhh.yaml` is the native PartiDesign scaffold definition, selected as
`antibody.scaffold: bindcraft2_vhh`. It includes `B1..127`; its three operational
edit windows are 27–33, 52–57 and 100–114. They are not an IMGT assignment. The
whole windows are excluded and reinserted at 27, 52 and 100 with final allowed
lengths 7, 5–7 and 3–25. Template coordinates are hidden for all three windows
and framework positions 35, 37, 45, 47, 50, 59, 97–99 and 115–116. Other framework
positions supply soft geometry guidance when `antibody.template: true`.

Template visibility, identity permissions and the intended paratope are separate
choices. The accompanying examples open the listed 11 framework positions for
identity design and declare a different framework selector for the paratope.
