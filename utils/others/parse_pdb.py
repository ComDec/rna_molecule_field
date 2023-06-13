import os
import numpy as np
from Bio.PDB.PDBParser import PDBParser


def read_pdb(pdb_file):
    p = PDBParser(PERMISSIVE=1)
    filename = pdb_file
    structure_id = os.path.basename(filename).split('.')[0]
    s = p.get_structure(structure_id, filename)
    model = s[0]
    chain = model.get_list()
    residue = chain[0].get_list()
    
    # parsing residues
    frames = []
    seq = ''
    for i in range(len(residue)):
        res_frames = []
        seq += str(residue[i].resname).strip(' ')
        atoms = residue[i]
        for atom in atoms:
            res_frames.append(atom.get_coord())
        frames.append(res_frames)
    frames = np.array(frames)
    if frames.shape[0] == len(residue) == len(seq):
        return frames, seq
    else:
        ValueError(f"Fatal error with mismatched length {frames.shape[0]} and {len(residue)}")
    
    
if __name__ == "__main__":
    import sys
    import pickle
    root_path = sys.argv[1]
    save_name = sys.argv[2]

    all_dict = []
    for idx, all_list in enumerate(os.listdir(root_path)):
        id = all_list.split(".")[0]
        file_path = os.path.join(root_path, all_list)
        fname, seq = read_pdb(file_path)
        all_dict.append(
            {idx: 
                {
                    "seq": seq,
                    "frame": fname
                }
            }
        )
    print(all_dict)
    with open(save_name, "wb+") as f:
        pickle.dump(all_dict, f)