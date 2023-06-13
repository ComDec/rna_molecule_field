import os
import jax
import wandb
from jax import jit, vmap
import jax.numpy as jnp
import openmm.app as app
app.Topology.loadBondDefinitions("RNA-top.xml")
import openmm.unit as unit
import numpy as np
import mdtraj as md
from tqdm import tqdm
from dmff.optimize import MultiTransform, genOptimizer
import optax
from torch.utils.data import Dataset, DataLoader
from dmff import Hamiltonian, NeighborListFreud

def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=True)
    return index

def compute_traj_list(trajectory, pdb_path, xml_path):
    hamilt = Hamiltonian(xml_path)
    top_pdb = app.PDBFile(pdb_path)
    pot = hamilt.createPotential(top_pdb.topology, nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff=1.0 * unit.nanometer)
    nbgen = [g for g in hamilt.getGenerators() if g.name == "NonbondedForce"][0]

    traj = md.load(trajectory, top=pdb_path)
    pos_list, box_list, pairs_list = [], [], []

    for frame in tqdm(traj):

        box = jnp.array([
            [10.0,  0.0,  0.0],
            [ 0.0, 10.0,  0.0],
            [ 0.0,  0.0, 10.0]
        ])
        positions = jnp.array(frame.xyz[0, :, :])
        nbobj = NeighborListFreud(box, 1.0, nbgen.covalent_map)
        nbobj.capacity_multiplier = 1
        pairs = nbobj.allocate(positions)
        box_list.append(box)
        pairs_list.append(pairs)
        pos_list.append(positions)

    pmax = max([p.shape[0] for p in pairs_list])
    pairs_jax = np.zeros((traj.n_frames, pmax, 3), dtype=int) + traj.n_atoms
    for nframe in range(traj.n_frames):
        pair = pairs_list[nframe]
        pairs_jax[nframe,:pair.shape[0],:] = pair[:,:]
    
    return np.array(pos_list), np.array(box_list), np.array(pairs_jax)

def custom_collect_fn(batch):
    fp_traj_ = np.array([batch[x]['fp_traj'] for x in range(len(batch))])
    pos_list_ = np.array([batch[x]['cg_traj']['pos_list'] for x in range(len(batch))])
    box_list_ = np.array([batch[x]['cg_traj']['box_list'] for x in range(len(batch))])
    pairs_jax_ = np.array([batch[x]['cg_traj']['pairs_jax'] for x in range(len(batch))])
    return {
            'fp_traj': fp_traj_,
            'cg_traj': {
                'pos_list': pos_list_,
                'box_list': box_list_,
                'pairs_jax': pairs_jax_
            }
        }


class traj_generator(Dataset):
    def __init__(self, params):
        self.fp_traj = np.loadtxt(params['fp_traj'], delimiter=',', dtype=str)[..., 0:-1].astype(float)
        self.xml = params['xml_path']
        self.pdb_path = params['pdb_path']
        self.cg_traj = params['cg_traj']
        self.sample_size = params['sample_size']
        self.pos_list, self.box_list, self.pairs_jax = compute_traj_list(self.cg_traj, self.pdb_path, self.xml)

    def __len__(self):
        return len(self.pos_list)

    def __getitem__(self, idx):
        return {
            'fp_traj': self.fp_traj[idx],
            'cg_traj': {
                'pos_list': self.pos_list[idx, ...],
                'box_list':self.box_list[idx, ...],
                'pairs_jax': self.pairs_jax[idx, ...]
            }
        }

class trainer:

    def __init__(self, params):
        self.md_dataset = traj_generator(params)
        self.md_dataloader = DataLoader(self.md_dataset, batch_size=params['sample_size'], collate_fn=custom_collect_fn, shuffle=True)
        self.params = params


    def loss_func(self, fp_traj, cg_traj, efunc, params):

        fp_traj = jnp.array(fp_traj)
        fp_energy = fp_traj[:, 1]

        cg_energy = vmap(efunc, in_axes=(0, 0, 0, None))(jnp.array(cg_traj['pos_list']), jnp.array(cg_traj['box_list']), jnp.array(cg_traj['pairs_jax']), params)

        beta = jnp.divide(1, 8.314e-3 * 300)
        delta = beta * (jnp.subtract(fp_energy, cg_energy))

        delta_mean = jnp.mean(delta)

        return jnp.log(jnp.mean(jnp.exp(delta - delta_mean - jnp.max(delta - delta_mean)))) + jnp.max(delta - delta_mean)


    def train(self):

        runtime_name = '{}_lr_k_{}_{}_LJ_{}_{}_prop_k_{}_{}_{}_prop_phase_{}_{}_{}_sample_size{}'.format(self.params['xml_path'],
            self.params['HB_k'], self.params['HA_k'], self.params['LJ_sig'], self.params['LJ_eps'],
            self.params['prop_k_1'],
            self.params['prop_k_2'], self.params['prop_k_3'], self.params['prop_phase_1'], self.params['prop_phase_2'],
            self.params['prop_phase_3'], self.params['sample_size']
        )

        wandb.init(project = self.params['project_name'], name=runtime_name, config=self.params)

        hamilt = Hamiltonian(self.params['xml_path'])
        top_pdb = app.PDBFile(self.params['pdb_path'])
        pot = hamilt.createPotential(top_pdb.topology, nonbondedMethod=app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.0 * unit.nanometer)
        efunc = pot.getPotentialFunc()

        multiTrans = MultiTransform(hamilt.paramtree)

        # multiTrans["HarmonicBondForce/k"] = genOptimizer(lrate=self.params['HB_k'], clip=0.005)
        # multiTrans["HarmonicAngleForce/k"] = genOptimizer(lrate=self.params['HA_k'], clip=0.005)

        # multiTrans["NonbondedForce/sigma"] = genOptimizer(lrate=self.params['nonBond_sig'], clip=0.005)
        # multiTrans["NonbondedForce/epsilon"] = genOptimizer(lrate=self.params['nonBond_eps'], clip=0.005)

        multiTrans["LennardJonesForce/sigma_nbfix"] = genOptimizer(lrate=self.params['LJ_sig'], clip=0.005)
        multiTrans["LennardJonesForce/epsilon_nbfix"] = genOptimizer(lrate=self.params['LJ_eps'], clip=0.005)

        multiTrans["PeriodicTorsionForce/prop_k/1"] = genOptimizer(lrate=self.params['prop_k_1'], clip=0.05)
        multiTrans["PeriodicTorsionForce/prop_k/2"] = genOptimizer(lrate=self.params['prop_k_2'], clip=0.05)
        multiTrans["PeriodicTorsionForce/prop_k/3"] = genOptimizer(lrate=self.params['prop_k_3'], clip=0.05)

        multiTrans["PeriodicTorsionForce/prop_phase/1"] = genOptimizer(lrate=self.params['prop_phase_1'], clip=0.05)
        multiTrans["PeriodicTorsionForce/prop_phase/2"] = genOptimizer(lrate=self.params['prop_phase_2'], clip=0.05)
        multiTrans["PeriodicTorsionForce/prop_phase/3"] = genOptimizer(lrate=self.params['prop_phase_3'], clip=0.05)

        multiTrans.finalize()
        grad_transform = optax.multi_transform(multiTrans.transforms, multiTrans.labels)
        opt_state = grad_transform.init(hamilt.paramtree)


        for loop in range(self.params['epoch']):
            print("LOOP: {}".format(loop))
            for sample in tqdm(self.md_dataloader):
                fp_traj = sample['fp_traj']
                cg_traj = sample['cg_traj']
                v, g = jax.value_and_grad(self.loss_func, 3)(fp_traj, cg_traj, efunc, hamilt.paramtree)
                wandb.log({'rel_entropy': v})
                updates, opt_state = grad_transform.update(g, opt_state, params=hamilt.paramtree)
                newprm = optax.apply_updates(hamilt.paramtree, updates)
                hamilt.updateParameters(newprm)
                if loop % 100 == 0:
                    hamilt.render(f"{runtime_name}loop-{loop}.xml")



if __name__ == '__main__':
    os.chdir('/home/MD/RNA')
    params = {
        'fp_traj': './data/AAAAAAGU/mol.out',
        'cg_traj': './data/AAAAAAGU/mol_cg.dcd',
        'pdb_path': './data/AAAAAAGU/mol_cg_fixed.pdb',
        'xml_path': './RNA.xml',
        'sample_size': 4000,
        'epoch': 200000,
        'project_name': 'CG_MD',
        'HB_k': 0,
        'HA_k': 0,
        'LJ_sig': 1e-3,
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