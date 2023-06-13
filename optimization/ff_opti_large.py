import os
import random
import string

import jax
import wandb
from jax import jit, vmap
import jax.numpy as jnp
import openmm.app as app
app.Topology.loadBondDefinitions("RNA-top-test.xml")
import openmm.unit as unit
import numpy as np
import mdtraj as md
from tqdm import tqdm
from dmff.optimize import MultiTransform, genOptimizer
import optax
import pandas as pd
from torch.utils.data import Dataset
from dmff import Hamiltonian, NeighborListFreud

def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=True)
    return index

def load_traj_energy(mol_out, params):

    df = pd.read_table(mol_out, delimiter=',')

    fp_traj = df['Potential Energy (kJ/mole)'].to_numpy(copy=True)[params['frame_start']: params['frame_end']].astype(float)

    return  fp_traj

def compute_traj_coords(trajectory, pdb_path, xml_path, cutoff, params):
    hamilt = Hamiltonian(xml_path)
    top_pdb = app.PDBFile(pdb_path)
    pot = hamilt.createPotential(top_pdb.topology, nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff= cutoff * unit.nanometer)
    nbgen = [g for g in hamilt.getGenerators() if g.name == "NonbondedForce"][0]

    traj = md.load(trajectory, top=pdb_path)[params['frame_start']: params['frame_end']]

    pos_list, box_list, pairs_list = [], [], []

    for frame in tqdm(traj):

        box = jnp.array([
            [10.0,  0.0,  0.0],
            [ 0.0, 10.0,  0.0],
            [ 0.0,  0.0, 10.0]
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

def custom_collect_fn(batch):
    fp_traj_ = np.array([batch[x]['fp_traj'] for x in range(len(batch))])
    cg_energy_ = jnp.array([batch[x]['cg_energy'] for x in range(len(batch))])
    return {
            'fp_traj': fp_traj_,
            'cg_energy': cg_energy_
        }


class traj_generator(Dataset):
    def __init__(self, params):
        self.xml = params['xml_path']

        traj_list = os.listdir(params['traj_folder'])
        cutoff = float(params['cutoff'])

        idx_ = getRandomIndex(len(traj_list), params['sample_size'])
        traj_list = np.array(traj_list)[idx_].tolist()

        hamilt = Hamiltonian(params['xml_path'])

        self.efunc_array = []
        self.fp_array = []
        self.pos_array= []
        self.box_array = []
        self.pair_array = []

        for idx in range(len(traj_list)):
            root_path = os.path.join(params['traj_folder'], traj_list[idx])
            fp_ = load_traj_energy(os.path.join(root_path, 'mol.out'), params)
            pos_, box_, pair_ = compute_traj_coords(os.path.join(root_path, 'mol_cg.dcd'), os.path.join(root_path, 'mol_cg_fixed.pdb'), self.xml, cutoff, params)
            top_pdb = app.PDBFile(os.path.join(root_path, 'mol_cg_fixed.pdb'))
            pot = hamilt.createPotential(top_pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff = cutoff * unit.nanometer)
            efunc = pot.getPotentialFunc()

            self.fp_array.append(fp_)
            self.pos_array.append(pos_)
            self.box_array.append(box_)
            self.pair_array.append(pair_)
            self.efunc_array.append(efunc)


    def __len__(self):
        return len(self.fp_array)

    def __getitem__(self, idx):
        return {
            'fp_traj': self.fp_array[idx],
            'cg_traj': {
                'pos_list': self.pos_array[idx],
                'box_list': self.box_array[idx],
                'pairs_jax': self.pair_array[idx]
            },
            'efunc': self.efunc_array[idx]
        }

class trainer:

    def __init__(self, params):
        self.params = params


    def loss_func(self, fp_traj, cg_traj, efunc, params):

        fp_energy = jnp.array(fp_traj)
        cg_energy = vmap(efunc, in_axes=(0, 0, 0, None))(jnp.array(cg_traj['pos_list']), jnp.array(cg_traj['box_list']), jnp.array(cg_traj['pairs_jax']), params)

        beta = jnp.divide(1, 8.314e-3 * 300)
        delta = beta * (jnp.subtract(fp_energy, cg_energy))

        delta_mean = jnp.mean(delta)

        return jnp.log(jnp.mean(jnp.exp(delta - delta_mean - jnp.max(delta - delta_mean)))) + jnp.max(delta - delta_mean)


    def train(self):

        random_name = ''.join(random.sample(string.ascii_letters + string.digits, 4))
        runtime_name = '{}_LJ_{}_{}_{}'.format(random_name, self.params['LJ_sig'], self.params['LJ_eps'], self.params['xml_name'])

        folder_name = '{}_k_{}_{}_LJ_{}_{}_prop_k_{}_{}_{}_prop_phase_{}_{}_{}_sample_size_{}_step_{}_cf_{}'.format(
            self.params['xml_name'], self.params['HB_k'], self.params['HA_k'], self.params['LJ_sig'], self.params['LJ_eps'],
            self.params['prop_k_1'],
            self.params['prop_k_2'], self.params['prop_k_3'], self.params['prop_phase_1'], self.params['prop_phase_2'],
            self.params['prop_phase_3'], self.params['sample_size'], self.params['update_step'], self.params['cutoff']
            )

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        wandb.init(project = self.params['project_name'], name=runtime_name, config=self.params)

        for loop in range(self.params['epoch']):

            hamilt = Hamiltonian(params['xml_path'])
            multiTrans = MultiTransform(hamilt.paramtree)

            # multiTrans["HarmonicBondForce/k"] = genOptimizer(lrate=self.params['HB_k'], clip=0.005)
            # multiTrans["HarmonicAngleForce/k"] = genOptimizer(lrate=self.params['HA_k'], clip=0.005)

            # multiTrans["NonbondedForce/sigma"] = genOptimizer(lrate=self.params['nonBond_sig'], clip=0.005)
            # multiTrans["NonbondedForce/epsilon"] = genOptimizer(lrate=self.params['nonBond_eps'], clip=0.005)

            multiTrans["LennardJonesForce/sigma_nbfix"] = genOptimizer(lrate=self.params['LJ_sig'], clip=0.005)
            multiTrans["LennardJonesForce/epsilon_nbfix"] = genOptimizer(lrate=self.params['LJ_eps'], clip=0.02)

            multiTrans["PeriodicTorsionForce/prop_k/1"] = genOptimizer(lrate=self.params['prop_k_1'], clip=0.05)
            multiTrans["PeriodicTorsionForce/prop_k/2"] = genOptimizer(lrate=self.params['prop_k_2'], clip=0.05)
            multiTrans["PeriodicTorsionForce/prop_k/3"] = genOptimizer(lrate=self.params['prop_k_3'], clip=0.05)

            multiTrans["PeriodicTorsionForce/prop_phase/1"] = genOptimizer(lrate=self.params['prop_phase_1'], clip=0.05)
            multiTrans["PeriodicTorsionForce/prop_phase/2"] = genOptimizer(lrate=self.params['prop_phase_2'], clip=0.05)
            multiTrans["PeriodicTorsionForce/prop_phase/3"] = genOptimizer(lrate=self.params['prop_phase_3'], clip=0.05)

            multiTrans.finalize()
            # grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
            grad_transform = optax.MultiSteps(optax.multi_transform(multiTrans.transforms, multiTrans.labels),
                                              every_k_schedule=self.params['update_step'], use_grad_mean=True)
            opt_state = grad_transform.init(hamilt.paramtree)

            md_dataset = traj_generator(self.params)
            idx_array = np.arange(self.params['sample_size'])

            for sub_loop in range(self.params['sub_loop']):
                np.random.shuffle(idx_array)
                for idx in tqdm(idx_array):
                    sample = md_dataset[idx]
                    fp_traj = sample['fp_traj']
                    sub_idx = getRandomIndex(len(fp_traj), self.params['frame_size']).tolist()

                    fp_traj = np.array([sample['fp_traj'][x,...] for x in sub_idx])
                    cg_traj = {
                        'pos_list': np.array([sample['cg_traj']['pos_list'][x,...] for x in sub_idx]),
                        'box_list': np.array([sample['cg_traj']['box_list'][x,...] for x in sub_idx]),
                        'pairs_jax': np.array([sample['cg_traj']['pairs_jax'][x,...] for x in sub_idx])
                    }
                    efunc = sample['efunc']

                    v, g = jax.value_and_grad(self.loss_func, 3)(fp_traj, cg_traj, efunc, hamilt.paramtree)
                    wandb.log({'rel_entropy': v})

                    updates, opt_state = grad_transform.update(g, opt_state, params=hamilt.paramtree)
                    newprm = optax.apply_updates(hamilt.paramtree, updates)
                    lg_sig = np.array(newprm['LennardJonesForce']['sigma_nbfix'])
                    wandb.log({f"lg_sig/pairs-{ii}": loss for ii, loss in enumerate(lg_sig)})
                    lg_eps = np.array(newprm['LennardJonesForce']['epsilon_nbfix'])
                    wandb.log({f"lg_eps/pairs-{ii}": loss for ii, loss in enumerate(lg_eps)})
                    hamilt.updateParameters(newprm)

                if sub_loop % 10 == 0:
                    hamilt.render(os.path.join(folder_name, f"loop-{loop}-sub-{sub_loop}.xml"))



if __name__ == '__main__':
    os.chdir('/home/MD/RNA')
    params = {
        'traj_folder': './data_v2',
        'xml_path': './RNA-test-new.xml',
        'xml_name': 'RNA-test',
        'frame_start': 2000,
        'frame_end': 4000,
        'sample_size': 2,
        'frame_size': 800,
        'cutoff': 1.2,
        'epoch': 3,
        'update_step': 2,
        'sub_loop': 5000,
        'project_name': 'CG_RNA',
        'HB_k': 0,
        'HA_k': 0,
        'LJ_sig': 1e-2,
        'LJ_eps': 1e-3,
        'nonBond_sig': 0,
        'nonBond_eps': 0,
        'prop_k_1': 1e-3,
        'prop_k_2': 1e-3,
        'prop_k_3': 1e-3,
        'prop_phase_1': 1e-3,
        'prop_phase_2': 1e-3,
        'prop_phase_3': 1e-3,

    }
    md_trainer = trainer(params)
    md_trainer.train()