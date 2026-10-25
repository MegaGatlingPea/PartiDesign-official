"""BC2 coordinate definitions adapted to explicit PD assemblies and CPU arrays.

Modifications: PD reader, full coordinate precision, explicit partners and finite
failures. See licenses/BindCraft2-LICENSE for the upstream terms and attribution.
"""
from __future__ import annotations
from pathlib import Path
import tempfile

MAX_SASA=dict(zip("ARNDCQEGHILKMFPSTWYV",(121.,265.,187.,187.,148.,214.,214.,97.,216.,195.,191.,230.,203.,228.,154.,143.,163.,264.,255.,165.)))
MASS=dict(zip("ARNDCQEGHILKMFPSTWYV",(71.08,156.19,114.10,115.09,103.14,128.13,129.12,57.05,137.14,113.16,113.16,128.17,131.19,147.18,97.12,87.08,101.10,186.21,163.18,99.13)))
PKA={"D":3.86,"E":4.25,"C":8.33,"Y":10.07,"H":6.,"K":10.53,"R":12.48}
HYDROPHOBIC=frozenset("ACVILMPFWY")

def _charge(sequence,ph):
    value=1/(1+10**(ph-9.69))-1/(1+10**(2.34-ph))
    for aa,pka in PKA.items():
        value+=sequence.count(aa)*(1/(1+10**(ph-pka)) if aa in "HKR" else -1/(1+10**(pka-ph)))
    return value

def _dssp(atoms,binder_chains,executable):
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    from Bio.PDB import PDBParser,DSSP
    if not executable: raise ValueError("DSSP executable is missing")
    chains=list(dict.fromkeys(atoms.chain_id.tolist()))
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if len(chains)>len(alphabet): raise ValueError("too many chains for the DSSP adapter")
    aliases=dict(zip(chains,alphabet)); private=atoms.copy()
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
    wanted=[]; numbers={chain:0 for chain in chains}
    for start,stop in zip(starts[:-1],starts[1:]):
        chain=str(atoms.chain_id[start]);numbers[chain]+=1
        private.res_id[start:stop]=numbers[chain]
        private.ins_code[start:stop]=""
        private.chain_id[start:stop]=aliases[chain]
        if chain in binder_chains: wanted.append((aliases[chain],(" ",numbers[chain]," ")))
    with tempfile.TemporaryDirectory(prefix="pd-postfilter-dssp-") as directory:
        path=Path(directory)/"input.pdb"
        file=PDBFile();file.set_structure(private);file.write(path)
        path.write_text("HEADER    POSTFILTER INPUT\n"+path.read_text())
        model=PDBParser(QUIET=True).get_structure("input",str(path))[0]
        assignments=DSSP(model,str(path),dssp=executable)
        if not wanted or any(key not in assignments for key in wanted): raise ValueError("DSSP did not assign every binder residue")
        states=[assignments[key][2] for key in wanted]
    helix=sum(s in "HGI" for s in states);sheet=sum(s=="E" for s in states)
    return {"bc1_binder_helix_percent":round(100*helix/len(states),2),
            "bc1_binder_sheet_percent":round(100*sheet/len(states),2),
            "bc1_binder_loop_percent":round(100*(len(states)-helix-sheet)/len(states),2)}

def compute_input_metrics(candidate,config):
    import numpy as np
    import biotite.structure as struc
    from scipy.spatial import cKDTree
    from src.design.utils.reader import read_structure
    from src.design.utils.atoms import PROTEIN_RES_ATOMS,PROTEIN_ATOM37
    from Bio.SeqUtils import seq1
    atoms=read_structure(Path(candidate.refold_path), canonical_polymer_order=True)
    from .records import snapshot_refold
    coordinates=np.asarray(snapshot_refold(candidate.refold_path)["coordinates"],dtype=np.float64)
    if coordinates.shape!=atoms.coord.shape or not np.array_equal(coordinates.astype(np.float32),atoms.coord):
        raise ValueError("source precision coordinates disagree with atom identities")
    heavy=~np.isin(atoms.element,("H","D","T"))
    atoms=atoms[heavy];coordinates=coordinates[heavy]
    # Ordered metric view; public atom identities and coordinates are unchanged.
    order=[]
    for chain in dict.fromkeys(atoms.chain_id.tolist()):
        chain_mask=atoms.chain_id==chain
        for uid in dict.fromkeys(atoms.residue_uid[chain_mask].tolist()):
            indices=np.flatnonzero(chain_mask&(atoms.residue_uid==uid))
            order.extend(sorted(indices,key=lambda i:str(atoms.atom_name[i])))
    atoms=atoms[order];coordinates=coordinates[order]
    if not len(atoms) or not np.isfinite(atoms.coord).all(): raise ValueError("empty or nonfinite selected refold")
    if any(n not in PROTEIN_RES_ATOMS or n=="UNK" for n in atoms.res_name): raise ValueError("BC2 requires exact standard protein residues")
    binder=np.isin(atoms.chain_id,candidate.binder_chains);target=np.isin(atoms.chain_id,candidate.target_chains)
    if not binder.any() or not target.any() or not (binder|target).all() or (binder&target).any(): raise ValueError("incomplete or overlapping protein partners")
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
    if len(starts)-1!=len(set(zip(atoms.chain_id.tolist(),atoms.residue_uid.tolist()))):
        raise ValueError("metric residue occurrences have ambiguous numbering")
    residue_id=np.repeat(np.arange(len(starts)-1),np.diff(starts))
    residue_chains=atoms.chain_id[starts[:-1]]
    binder_res=np.flatnonzero(np.isin(residue_chains,candidate.binder_chains))
    sequence=np.array([seq1(n,undef_code="?") for n in atoms.res_name[starts[:-1]]])
    if any(a not in MAX_SASA for a in sequence): raise ValueError("unknown standard sequence")
    metrics,errors,details={},{},{"coordinate_stage":"input_success_refold","sasa_probe_radius":1.4,"sasa_points":100}
    for name,mask in (("backbone_clashes",atoms.atom_name=="CA"),("all_atom_clashes",np.isin(atoms.atom_name,PROTEIN_ATOM37))):
        subset=atoms[mask]
        pairs=np.array(list(cKDTree(coordinates[mask]).query_pairs(2.5)),dtype=int).reshape(-1,2)
        metrics["bc2_"+name]=int(np.sum(subset.chain_id[pairs[:,0]]!=subset.chain_id[pairs[:,1]]))
    contacts=cKDTree(coordinates[binder]).query_ball_tree(cKDTree(coordinates[target]),4.)
    binder_indices=np.flatnonzero(binder);target_indices=np.flatnonzero(target)
    contacted={residue_id[binder_indices[i]] for i,indices in enumerate(contacts) if indices}
    contact_mask=np.array([rid in contacted for rid in binder_res])
    metrics["bc2_interface_residues"]=int(contact_mask.sum())
    binder_sequence=sequence[binder_res]
    for aa in MAX_SASA: metrics["bc2_interface_aa_"+aa]=int(np.sum(contact_mask&(binder_sequence==aa)))
    metrics["bc2_receptor_chains_contacted"]=len({str(atoms.chain_id[target_indices[j]]) for indices in contacts for j in indices})
    hydrophobic=np.isin(binder_sequence,list(HYDROPHOBIC))
    if contact_mask.any(): metrics["bc2_interface_hydrophobicity"]=float(np.mean(hydrophobic[contact_mask]))
    else: errors["bc2_interface_hydrophobicity"]="no binder-target contacts"
    def residue_sasa(array):
        try: per_atom=struc.sasa(array,probe_radius=1.4,point_number=100)
        except KeyError:
            details["sasa_radii"]="Single (ProtOr unavailable)"
            per_atom=struc.sasa(array,probe_radius=1.4,point_number=100,vdw_radii="Single")
        if not np.isfinite(per_atom).all(): raise ValueError("nonfinite solvent accessible area")
        return struc.apply_residue_wise(array,per_atom,np.sum)
    bound=maximum=None
    try:
        free=residue_sasa(atoms[binder]);bound=residue_sasa(atoms)[binder_res]
        maximum=np.array([MAX_SASA[a] for a in binder_sequence]);exposed=free/maximum>=.2
        metrics["bc2_surface_hydrophobicity"]=float(hydrophobic[exposed].mean()) if exposed.any() else 0.
        metrics["bc2_interface_buried_area"]=max(0.,float(free.sum()-bound.sum()))
        if free.sum()<=0: raise ValueError("zero isolated binder surface")
        metrics["bc2_interface_buried_area_fraction"]=max(0.,float((free.sum()-bound.sum())/free.sum()))
        terminal=[];offset=0
        for chain in dict.fromkeys(residue_chains[binder_res].tolist()):
            n=int(np.sum(residue_chains[binder_res]==chain))
            terminal.extend(offset+np.r_[np.clip(np.arange(3),0,n-1),np.clip(n-1-np.arange(3),0,None)]);offset+=n
        metrics["bc2_terminus_exposure"]=float((bound/maximum)[terminal].mean())
    except Exception as exc:
        for name in ("surface_hydrophobicity","interface_buried_area","interface_buried_area_fraction","terminus_exposure","exposed_loop_fraction"):
            errors["bc2_"+name]=str(exc)
            metrics.pop("bc2_"+name,None)
        bound=maximum=None
    try:
        states=np.concatenate([struc.annotate_sse(atoms[atoms.chain_id==c]) for c in dict.fromkeys(residue_chains[binder_res].tolist())])
        for label,mask in (("helix",states=="a"),("beta_sheet",states=="b"),("loop",~np.isin(states,("a","b")))):
            metrics["bc2_binder_"+label+"_fraction"]=float(mask.mean())
            if contact_mask.any(): metrics["bc2_interface_"+label+"_fraction"]=float(mask[contact_mask].mean())
        loops=~np.isin(states,("a","b"))
        if bound is not None: metrics["bc2_exposed_loop_fraction"]=float(np.mean((bound/maximum>=.2)[loops])) if loops.any() else 0.
    except Exception as exc:
        for label in ("helix","beta_sheet","loop"): errors["bc2_binder_"+label+"_fraction"]=str(exc)
    breaks=0;observed=False
    for chain in candidate.binder_chains:
        ca=coordinates[(atoms.chain_id==chain)&(atoms.atom_name=="CA")]
        if len(ca)>1:
            breaks+=int(np.sum(np.linalg.norm(np.diff(ca,axis=0),axis=1)>4.3));observed=True
    if observed: metrics["bc2_binder_chain_breaks"]=breaks
    else: errors["bc2_binder_chain_breaks"]="no adjacent resolved CA pair"
    text="".join(binder_sequence)
    metrics.update(bc2_binder_length=len(text),bc2_binder_mass_kda=(sum(MASS[a] for a in text)+18.02)/1000,
                   bc2_binder_cysteines=text.count("C"),bc2_binder_net_charge=_charge(text,7.4))
    lo,hi=0.,14.
    for _ in range(50):
        mid=(lo+hi)/2
        if _charge(text,mid)>0: lo=mid
        else: hi=mid
    metrics["bc2_binder_pi"]=(lo+hi)/2
    details["sequence_properties"]="concatenated assembly; BC2 single-terminal-pair convention"
    try: metrics.update(_dssp(atoms,candidate.binder_chains,config.get("dssp")))
    except Exception as exc: errors["bc1_binder_loop_percent"]=str(exc)
    for name,value in list(metrics.items()):
        if not np.isfinite(value): errors[name]="nonfinite metric";del metrics[name]
    return {"metrics":metrics,"errors":errors,"details":details}
