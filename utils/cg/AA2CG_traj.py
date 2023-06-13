import argparse
import json
import codecs
import os

import numpy as np
import mdtraj as md

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_aa', type=str, default='./run_500ns.dcd')
    parser.add_argument('--traj_cg', type=str, default='./run_cg_500ns.dcd')
    parser.add_argument('--pdb_aa', type=str, default='./ala_solv.pdb')
    parser.add_argument('--pdb_cg', type=str, default='./cg_ala.pdb')
    parser.add_argument('--mapping_mode', type=str, default='com')
    parser.add_argument('--slice', type=str, default='./ala_slice.json')
    parser.add_argument('--mass', type=str, default='./ala_mass.json')

    opt = parser.parse_args()


    with codecs.open(opt.slice, 'r', 'utf-8') as f:
        atom_slice = json.load(f)

    with codecs.open(opt.mass, 'r', 'utf-8') as f:
        atom_mass = json.load(f)

    traj_aa = md.load(opt.traj_aa, top=opt.pdb_aa)
    traj_cg = md.load(opt.pdb_cg)

    atom_nums = sum([len(atom_slice[x].keys()) for x in range(len(atom_slice))])
    cg_coords = np.zeros([traj_aa.xyz.shape[0], atom_nums, 3], dtype=np.float32)

    # parsing frames
    for i in range(traj_aa.xyz.shape[0]):
        aa_coords = traj_aa.xyz[i]

        atom_idx = 0
        # parsing residues
        for j in range(len(atom_slice)):

            # parsing atoms
            for key in atom_slice[j].keys():

                temp_slice = atom_slice[j][key]
                temp_mass = np.array(atom_mass[j][key]).reshape([-1])

                if opt.mapping_mode == 'com':
                    cg_coords[i][atom_idx][:] = cal_mass_center(aa_coords[temp_slice,:], temp_mass)
                    atom_idx += 1
                else:
                    cg_coords[i][atom_idx][:] = cal_geo_center(aa_coords[temp_slice,:])
                    atom_idx += 1

        assert atom_idx == atom_nums
        
    traj_cg.xyz = cg_coords
    traj_cg.save(opt.traj_cg)