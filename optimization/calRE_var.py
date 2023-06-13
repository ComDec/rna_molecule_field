import os
import pickle

import jax
from jax import jit, vmap
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import numpy as np
import mdtraj as md
from tqdm import tqdm

app.Topology.loadBondDefinitions("RNA-top.xml")
from dmff import Hamiltonian, NeighborListFreud

def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=True)
    return index


def compute_traj_list(trajectory, pdb_path, xml_path, frame_start, cutoff):
    hamilt = Hamiltonian(xml_path)
    top_pdb = app.PDBFile(pdb_path)
    cutoff = float(cutoff)
    pot = hamilt.createPotential(top_pdb.topology, nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff= cutoff * unit.nanometer)
    nbgen = [g for g in hamilt.getGenerators() if g.name == "NonbondedForce"][0]

    traj = md.load(trajectory, top=pdb_path)[frame_start:]
    pos_list, box_list, pairs_list = [], [], []

    for frame in tqdm(traj):
        box = jnp.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0]
        ])
        positions = jnp.array(frame.xyz[0, :, :])
        nbobj = NeighborListFreud(box, cutoff, nbgen.covalent_map)
        nbobj.capacity_multiplier = 1
        pairs = nbobj.allocate(positions)
        box_list.append(box)
        pairs_list.append(pairs)
        pos_list.append(positions)

    pmax = max([p.shape[0] for p in pairs_list])
    pairs_jax = np.zeros((traj.n_frames, pmax, 3), dtype=int) + traj.n_atoms
    for nframe in range(traj.n_frames):
        pair = pairs_list[nframe]
        pairs_jax[nframe, :pair.shape[0], :] = pair[:, :]

    return np.array(pos_list), np.array(box_list), np.array(pairs_jax)


def loss_func(fp_traj, cg_traj, efunc, params):

    fp_traj = jnp.array(fp_traj)
    fp_energy = fp_traj[:, 2]
    # T = fp_traj[:, 2]

    cg_energy = vmap(efunc, in_axes=(0, 0, 0, None))(jnp.array(cg_traj['pos_list']), jnp.array(cg_traj['box_list']),
                                                     jnp.array(cg_traj['pairs_jax']), params)

    beta = jnp.divide(1, 8.314e-3 * 300)
    delta = beta * (jnp.subtract(fp_energy, cg_energy))

    delta_mean = jnp.mean(delta)

    return jnp.log(jnp.mean(jnp.exp(delta - delta_mean - jnp.max(delta - delta_mean)))) + jnp.max(
        delta - delta_mean)

def train(params):

    hamilt = Hamiltonian(params['xml_path'])
    top_pdb = app.PDBFile(params['pdb_path'])
    pot = hamilt.createPotential(top_pdb.topology, nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff=1.0 * unit.nanometer)
    efunc = pot.getPotentialFunc()

    if params['seq_len'] == 6:
        fp_traj = np.loadtxt(params['fp_traj'], delimiter=',', dtype=str)[1100:, 0:-1].astype(float)
        cg1, cg2, cg3 = compute_traj_list(params['cg_traj_aa'], params['pdb_path'], params['xml_path'], 1100, 1.0)
    else:
        fp_traj = np.loadtxt(params['fp_traj'], delimiter=',', dtype=str)[2000:, 0:-1].astype(float)
        cg1, cg2, cg3 = compute_traj_list(params['cg_traj_aa'], params['pdb_path'], params['xml_path'], 2000, 1.0)

    cg_traj_aa = {
        'pos_list': cg1,
        'box_list': cg2,
        'pairs_jax': cg3
    }
    re_loss_aa = loss_func(fp_traj, cg_traj_aa, efunc, hamilt.paramtree)
    return re_loss_aa



if __name__ == '__main__':
    os.chdir('/home/MD/RNA')
    with open('re_record.pkl', 'rb') as f:
        d1 = pickle.load(f)

    with open('re_record_val.pkl', 'rb') as f:
        d2 = pickle.load(f)

    seq_list_ = os.listdir('./data')
    seq_list = []

    for i in seq_list_:
        if len(i) == 6:
            seq_list.append(i)
        else:
            continue

    re_list = []
    re_list_train = []
    for _ in tqdm(range(10)):
        for seq in np.random.choice(np.array(seq_list), 1).tolist():
            root_path = os.path.join('./data', seq)
            fp_path = os.path.join(root_path, 'mol.out')
            cg_path = os.path.join(root_path, 'mol_cg.dcd')
            pdb_path = os.path.join(root_path, 'mol_cg_fixed.pdb')
            params_1 = {
                'seq_len': len(seq),
                'fp_traj': fp_path,
                'cg_traj_aa': cg_path,
                'pdb_path': pdb_path,
                'xml_path': './RNA-test.xml',
            }

            params_2 = {
                'seq_len': len(seq),
                'fp_traj': fp_path,
                'cg_traj_aa': cg_path,
                'pdb_path': pdb_path,
                'xml_path': './RNA-test-train.xml',
            }
            re_loss = train(params_1)
            re_list.append(re_loss)

            re_loss_t = train(params_2)
            re_list_train.append(re_loss_t)

    with open('re_record.pkl', 'wb+') as f:
        pickle.dump(re_list, f)

    with open('re_record_val.pkl', 'wb+') as f:
        pickle.dump(re_list_train, f)