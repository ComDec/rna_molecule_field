import mdtraj as md
import os 
import numpy as np

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


def cal_unmatched_rmsd(target_pdb, ref_pdb):
    ref = md.load(ref_pdb)
    target = md.load(target_pdb)

    target_res_list = target.topology._residues
    ref_res_list = ref.topology._residues

    target_atom_slice = []
    ref_atom_slice = []

    for idx, residues in enumerate(target_res_list):
        atoms_list = residues._atoms
        ref_atoms_list = ref_res_list[idx]._atoms
        ref_atoms_dict = {}

        for atom in ref_atoms_list:
            ref_atoms_dict.update(
                {atom.name: (atom.index)})

        for atom in atoms_list:
            target_atom_name = atom.name
            target_atom_index = atom.index
            try:
                ref_atom_slice.append(ref_atoms_dict[target_atom_name])
                target_atom_slice.append(target_atom_index)
            except KeyError:
                continue

    return md.rmsd(target, ref, atom_indices=np.array(target_atom_slice), ref_atom_indices=np.array(ref_atom_slice))
