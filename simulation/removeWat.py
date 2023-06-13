import os

import mdtraj as md
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--save', type=str)

    opt = parser.parse_args()
    try:
        os.mkdir(opt.save)
    except:
        print('Dir Exist')

    traj = md.load(opt.traj, top=opt.top)
    aidx = [a.index for a in traj.topology.atoms if a.residue.name != "HOH"]
    traj_now = traj.atom_slice(aidx)
    frame = traj_now[0]
    frame.save(f"./{opt.save}/mol.pdb")
    traj_now.save(f"./{opt.save}/mol.dcd")

