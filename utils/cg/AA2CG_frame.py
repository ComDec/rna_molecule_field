import argparse
import json
import codecs
import numpy as np
import yaml
from Bio.PDB.PDBParser import PDBParser
import warnings
warnings.filterwarnings("ignore")

def convert_yaml_dict(yaml_path):
    with codecs.open(yaml_path, 'r', 'utf-8') as f_yaml:
        yaml_mapping = yaml.safe_load(f_yaml)

    json_dict = {}
    for item in yaml_mapping:
        top_keys = item['from_residue']
        sec_key = item['to_residue']
        yaml_atom = item['atom_mapping']
        for key in top_keys:
            temp_dict = {
                key: {
                    sec_key: yaml_atom
                }
            }
            json_dict.update(temp_dict)

    return json_dict

def parser_pdb_inline(atom_idx, atomname, residue_idx, residue_name, coords):
    """
    :param atom_idx: serial number of atoms
    :param atomname: name of atoms
    :param residue_idx: serial number of residue
    :param residue_name: name of residues
    :param coords: numpy.array object with shape: (3,)

    :return: A naive line of pdb file, which can be directly write in pdb file
            if you parse pdb file via "readlines()", you will obtain exactly the same result
    """

    # follow the pdb standard fe.3
    coords = np.around(coords, 3)

    inline = list(' ' * 80 + '\n')
    inline[0:4] = 'ATOM'
    inline[6:11] = str(atom_idx).rjust(5)
    inline[12:16] = str(atomname).ljust(4)
    inline[17:20] = str(residue_name).rjust(3)
    inline[22:26] = str(residue_idx).rjust(4)
    inline[30:38] = str(coords[0]).rjust(8)
    inline[38:46] = str(coords[1]).rjust(8)
    inline[46:54] = str(coords[2]).rjust(8)
    inline[54:60] = str('0.00').rjust(6)
    inline[60:66] = str('0.00').rjust(6)
    aline = "".join(inline)
    return aline

def cal_geo_center(coords):
    """

    :param coords: numpy.array object with shape: (N, 3) where N is the number of atoms constitute geo center
    :return: numpy.array object with shape (3, )
    """
    nums = coords.shape[0]
    return np.sum(coords, axis=0) / nums

def cal_mass_center(coords, mass):
    """

    :param coords: numpy.array object with shape: (N, 3) where N is the number of atoms constitute com center
    :param mass: numpy.array object with shape: (N, )
    :return: numpy.array object with shape (3, )
    """
    weighted_coords = np.einsum('ij, i -> ij', coords, mass)
    return np.sum(weighted_coords, axis=0) / np.sum(mass)

def AA2CG(pdb_aa, pdb_cg, map_file, slice, mass, mapping_mode):
    try:
        with codecs.open(map_file, 'r', 'utf-8') as f:
            mapping = json.load(f)
    except:
        mapping = convert_yaml_dict(map_file)

    # you can choose 'com' i.e. center of mass or 'geo' i.e. center of geometry
    try:
        mode = mapping_mode
    except KeyError:
        mode = 'com'

    # obtain residue by biopython, ignore multi chain/model structure
    p = PDBParser(PERMISSIVE=1)
    filename = pdb_aa
    structure_id = filename.split('.')[0]
    s = p.get_structure(structure_id, filename)
    model = s[0]
    chain = model.get_list()
    residue = chain[0].get_list()

    if pdb_cg is None:
        output_file = 'cg_' + pdb_aa
    else:
        output_file = pdb_cg

    with open(output_file, 'w+') as f:

        # Auto Mapping
        atom_idx = 0
        mapping_atom_slice = []
        mapping_atom_mass = []
        mapping_atom_coords = []
        sequences = []

        # parsing residues
        for i in range(len(residue)):
            atom = residue[i]
            residue_name = residue[i].resname
            base_list = ["A", "U", "C", "G"]

            if len(residue_name) > 1:
                extracted_chars = [char for char in residue_name if char in base_list]
                if len(extracted_chars) != 0:
                    residue_name = extracted_chars[0]
                else:
                    residue_name = 'UNK'
                
            # if i == 0 and len(residue_name) == 1:
            #     residue_name = residue[i].resname + '5'
            # if i == (len(residue) - 1) and len(residue_name) == 1:
            #     residue_name = residue[i].resname + '3'

            sequences.append(residue_name)
            try:
                cg_residue_name = [x for x in mapping[residue_name].keys()][0]

            except KeyError:
                print("{} residues name is not defined in mapping files at position {}".format(residue_name, i))
                continue

            residue_mapping = mapping[residue_name][cg_residue_name]
            atom_slice_dict = mapping[residue_name][cg_residue_name].copy()
            atom_mass_dict = mapping[residue_name][cg_residue_name].copy()
            atom_coords_dict = mapping[residue_name][cg_residue_name].copy()

            # parsing mapping dicts
            for keys in residue_mapping.keys():
                temp_atom_slice = []
                mapping_atoms = residue_mapping[keys]
                mapping_nums = len(mapping_atoms)

                # check for validation, ignore missing atoms in AA structures
                fix_mapping_atoms = []
                for sub_idx in range(mapping_nums):
                    try:
                        temp_atom = atom[mapping_atoms[sub_idx]]
                        fix_mapping_atoms.append(mapping_atoms[sub_idx])
                    except KeyError:
                        print('missing {} at residue {}, current {}'.format(mapping_atoms[sub_idx], i, residue_name))

                # new mapping atoms
                mapping_atoms = fix_mapping_atoms
                mapping_nums = len(mapping_atoms)
                temp_coords = np.ones([mapping_nums, 3])
                temp_mass = np.ones([mapping_nums])

                # calculate coords of CG beads
                for sub_idx in range(mapping_nums):
                    temp_atom = atom[mapping_atoms[sub_idx]]
                    temp_atom_slice.append(temp_atom.serial_number - 1)
                    temp_coords[sub_idx] = temp_atom.get_coord()
                    temp_mass[sub_idx] = temp_atom.mass

                atom_slice_dict[keys] = temp_atom_slice
                atom_mass_dict[keys] = temp_mass.tolist()
                atom_coords_dict[keys] = cal_geo_center(temp_coords).tolist()

                if temp_coords.shape[0] > 0:
                    # mapping mode
                    if mode == 'com':
                        weight_coords = cal_mass_center(temp_coords, temp_mass)
                    else:
                        weight_coords = cal_geo_center(temp_coords)
                    atom_idx += 1


                    # write in pdb file
                    pdb_inline = parser_pdb_inline(atom_idx, keys, i+1, cg_residue_name, weight_coords)
                    # print(pdb_inline)
                    f.write(pdb_inline)
                else:
                    del atom_slice_dict[keys]
                    del atom_mass_dict[keys]
                    del atom_coords_dict[keys]

                # else there's no atoms for CG beads

            mapping_atom_slice.append(atom_slice_dict)
            mapping_atom_mass.append(atom_mass_dict)
            mapping_atom_coords.append(atom_coords_dict)

        with codecs.open(slice, 'w+', 'utf-8') as f1:
            json.dump(mapping_atom_slice, f1)

        with codecs.open(mass, 'w+', 'utf-8') as f1:
            json.dump(mapping_atom_mass, f1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_aa', type=str, default='./evaluation/test.pdb')
    parser.add_argument('--pdb_cg', type=str, default='./evaluation/test_cg.pdb', help='by default will add cg_ to the input name')
    parser.add_argument('--slice', type=str, default='./ala_slice.json')
    parser.add_argument('--mass', type=str, default='./ala_mass.json')
    parser.add_argument('--map_file', type=str, default='mapping_rna.yml', help='mapping file in json or yaml, key represent CG atoms while values represent AA atoms for mapping')
    parser.add_argument('--mapping_mode', type=str, default='geo', help='geo or com mode')
    opt = parser.parse_args()

    AA2CG(pdb_aa=opt.pdb_aa, pdb_cg=opt.pdb_cg, map_file=opt.map_file, slice=opt.slice, mass=opt.mass, mapping_mode=opt.mapping_mode)
