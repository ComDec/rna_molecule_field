import argparse
import math
import os.path
import sys
from collections import Counter
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('/home/MD/RNA')
from utils.auto_logger import Logger
from utils.AA2CG_frame import AA2CG
from utils.tools import DotDict, cal_unmatched_rmsd
from remapping import targetFold, getAA_idx_from_CG
from generate_reference import genCGReference
import mdtraj as md
import numpy as np


def cg_rmsd(cg_traj_path, cg_top_path, cg_ref_path, save_folder, logger):

    logger.cri(f'Load Refercence CG Structure at {cg_ref_path}')
    logger.cri(f'Load CG Structure Toplogy at {cg_top_path}')
    logger.cri(f'Load CG Structure Traj at {cg_traj_path}')

    top1_save_path = os.path.join(save_folder, 'CG_BEST.pdb')
    traj = md.load(cg_traj_path, top=cg_top_path)
    ref_cg = md.load(cg_ref_path)

    logger.info(f'Traj at {traj}')
    logger.info(f'Reference CG at {ref_cg}')
    logger.cri('Calculate CG-Level RMSD')
    rmsd = md.rmsd(traj, ref_cg) * 10 / (-1.3 + math.log(traj.n_residues))
    naive_rmsd = md.rmsd(traj, ref_cg) * 10

    plt.hist(rmsd)
    hist_path = os.path.join(save_folder, os.path.basename(cg_ref_path).split('.')[0] + '.png')
    plt.title('CG-Level RMSD')
    plt.savefig(hist_path)

    logger.info('Saving RMSD Distribution in {}'.format(hist_path))
    top_rmsd = np.sort(rmsd)[:int(len(rmsd) * 0.01) + 1]
    top1_rmsd = np.sort(naive_rmsd)[0]
    top1_index = np.where(naive_rmsd == top1_rmsd)
    top1_traj = traj[top1_index]
    top1_traj.save(top1_save_path)

    logger.info('CG-Level RMSD Calculation Finished')
    logger.info('CG-Level Top 1% RMSD100 {}'.format(top_rmsd))
    logger.info('Saving Best CG Structure to {}'.format(top1_save_path))

    logger.cri('CG-Level Lowest RMSD {}'.format(top1_rmsd))
    logger.cri('Average CG-Level Top 1% RMSD100 {}å'.format(np.mean(top_rmsd)))

    return top1_save_path

def aa_rmsd(refined_structure_list, ref_aa_path, logger):
    ref_aa = md.load(ref_aa_path)
    logger.cri('Calculate AA-Level RMSD')
    rmsd_list = []
    naive_rmsd_list = []
    for refined_aa in refined_structure_list:
        pred_aa = md.load(refined_aa)

        try:
            rmsd = md.rmsd(pred_aa, ref_aa) * 10 / (-1.3 + math.log(pred_aa.n_residues))
            naive_rmsd = md.rmsd(pred_aa, ref_aa) * 10
        except:
            logger.war(f'Mismatch Structure, Please Check Your Label Structure {ref_aa}')
            logger.war(f'Mismatch Structure, Please Check Your Label Structure {pred_aa}')
            logger.info(f'Trying to Select Atoms According to {pred_aa}')

            rmsd = cal_unmatched_rmsd(target_pdb=refined_aa, ref_pdb=ref_aa_path)
            naive_rmsd = rmsd * 10
            rmsd = rmsd * 10 / (-1.3 + math.log(pred_aa.n_residues))
            

        logger.info('Structure AA-Level RMSD100 Catched {}'.format(rmsd))
        logger.info('Structure AA-Level RMSD Catched {}'.format(naive_rmsd))
        rmsd_list.append(rmsd)
        naive_rmsd_list.append(naive_rmsd)

    top_rmsd_100 = np.sort(rmsd_list)[0]
    top_rmsd = np.sort(naive_rmsd_list)[0]
    logger.cri('AA-Level RMSD Calculation Finished')

    logger.cri('AA-Level BEST RMSD100 {}'.format(top_rmsd_100))
    logger.cri('Average AA-Level Top 1% RMSD100 {}å'.format(np.mean(rmsd_list)))

    logger.cri('AA-Level BEST RMSD {}'.format(top_rmsd))
    logger.cri('Average AA-Level Top 1% RMSD {}å'.format(np.mean(naive_rmsd_list)))

def evaluate_energy_cluster(cg_traj_path, cg_top_path, cg_out_path, ref_cg_path, save_folder, logger):
    cg_id = os.path.basename(cg_traj_path).split('.')[0]
    traj_md = md.load(cg_traj_path, top=cg_top_path)
    logger.info('Energy Cluster Evaluation Step')
    logger.info(f'Load {len(traj_md)} Frames')
    distance_matrix = np.zeros([len(traj_md), len(traj_md)])

    try:
        potential_energy = pd.read_table(cg_out_path, delimiter=',')['Potential Energy (kJ/mole)'].to_numpy()
    except:
        potential_energy = pd.read_csv(cg_out_path, delimiter=',')['Potential Energy (kJ/mole)'].to_numpy()

    percent = 0
    for i in range(len(traj_md)):
        distance_matrix[i, :] = md.rmsd(traj_md, traj_md[i])
        percent = i / len(traj_md) * 100
        if percent % 1 == 0:
            logger.info(f'Distance Matrix Calculated {percent}%')
        if percent % 25 == 0:
            logger.cri(f'Distance Matrix Calculated {percent}%')
    logger.cri('Distance Matrix Calculated Finished')

    logger.cri('Start Traj Cluster Step')
    cluster = hdbscan.HDBSCAN(metric='precomputed')
    cluster.fit(distance_matrix)
    unique, counts = np.unique(cluster.labels_, return_counts=True)
    logger.cri(f'Cluster Finished with {len(unique)} Classes')
    summary_ = dict(zip(unique, counts))
    energy_summary_ = dict(zip(unique, counts))
    for cluster_id in energy_summary_.keys():
        idx = np.where(cluster.labels_ == cluster_id)[0]
        temp_energy_list = [potential_energy[x] for x in idx]
        energy_summary_[cluster_id] = np.mean(temp_energy_list)
    summary = Counter(summary_).most_common()
    energy_sorted = Counter(energy_summary_).most_common()

    if len(energy_sorted) >= 10:
        energy_summary = energy_sorted[-11:-1]
    else:
        energy_summary = energy_sorted

    ref = md.load(ref_cg_path)
    cluster_traj_path = os.path.join(save_folder, 'cluster')
    if not os.path.isdir(cluster_traj_path):
        os.mkdir(cluster_traj_path)

    traj_list_path = []

    for idx, group in enumerate(energy_summary):
        label = group[0]
        atom_slice = np.array(np.where(cluster.labels_ == label)).reshape([-1])
        traj = md.load(cg_traj_path, top=cg_top_path)[atom_slice]
        rmsd = md.rmsd(traj, ref) * 10 / (-1.3 + math.log(traj.n_residues))
        logger.info(f'Saving CG Traj Cluster {label} with Energy: {group[1]}, min RMSD: {np.min(rmsd)}, mean RMSD: {np.mean(rmsd)}')
        traj_save_path = os.path.join(cluster_traj_path, f'{cg_id}_{label}_rmsd_{np.mean(rmsd)}_E_{group[1]}.dcd')
        traj.save(traj_save_path)
        traj_list_path.append(traj_save_path)

    return traj_list_path

def cal_average_structure(traj_list_path, cg_top_path, save_folder, logger):
    logger.info('Calculate Average Structures')
    avg_structure_save_folder = os.path.join(save_folder, 'cluster_avg_strucutre')
    avg_structure_list_path = []
    log_save_folder = os.path.join(save_folder, 'gmx_log')

    if not os.path.isdir(avg_structure_save_folder):
        os.mkdir(avg_structure_save_folder)
    if not os.path.isdir(log_save_folder):
        os.mkdir(log_save_folder)

    for traj_path in traj_list_path:

        logger.info('Structure {}'.format(traj_path))
        traj_base_name = os.path.basename(traj_path).split('.')[0]
        avg_structure_save_path = os.path.join(avg_structure_save_folder, traj_base_name +'-AVG.pdb')
        eigenval_save_path = os.path.join(log_save_folder, f'eigenval_{traj_base_name}.xvg')
        eigenvec_save_path = os.path.join(log_save_folder, f'eigenvec_{traj_base_name}.trr')
        log_save_path = os.path.join(log_save_folder, f'{traj_base_name}.log')
        
        os.system('mdconvert -o {} {}'.format(traj_path+'.xtc', traj_path))
        os.system('printf \'0\n0\' | gmx covar -f {} -s {} -av {} -o {} -v {} -l {}'.format(traj_path+'.xtc', cg_top_path, avg_structure_save_path, eigenval_save_path, eigenvec_save_path, log_save_path))
        os.system('rm -rf {}'.format(traj_path+'.xtc'))

        avg_structure_list_path.append(avg_structure_save_path)

    logger.cri('Average Structure Calculation Finished')
    logger.cri(f'{len(avg_structure_list_path)} Structure Catched')
    return avg_structure_list_path

def remapping(naive_aa_path, avg_strucutre_list_path, save_folder, mapping_file, logger):
    naive_aa_path += '.pdb'
    refined_cluster_structure_path = os.path.join(save_folder, 'refined_structures')
    refined_cluster_structure_list = []

    if not os.path.isdir(refined_cluster_structure_path):
        os.mkdir(refined_cluster_structure_path)

    logger.cri('Start Remapping STEP')
    for avg_structure_path in avg_strucutre_list_path:
        structure_id = os.path.basename(avg_structure_path).split('.')[0]
        refined_aa_save_path = os.path.join(refined_cluster_structure_path, structure_id+'_refined.pdb')
        logger.info(f'Remapping {structure_id} to {refined_aa_save_path}')
        atoms_idx, group_com, sequence = getAA_idx_from_CG(aa_pdb=naive_aa_path, cg_pdb=avg_structure_path, mapping_file=mapping_file)
        targetFold(ref_pdb=naive_aa_path, atom_idx=atoms_idx, group_COM=group_com, save_path=refined_aa_save_path)
        refined_cluster_structure_list.append(refined_aa_save_path)
    logger.cri('Remapping STEP Finished')

    return refined_cluster_structure_list

def main(configs, logger):
    logger.info('Start Evaluation Processing')
    logger.debug('Checking Input File {}, {}, {}'.format(configs.cg_traj, configs.cg_out, configs.ref_pdb))

    try:
        temp_traj = md.load(configs.ref_pdb)
    except FileNotFoundError:
        logger.error('Missing File {}'.format(configs.ref_pdb))
        exit()

    # Generate Naive AA Structure STEP
    case_id = os.path.basename(configs.cg_traj).split('.')[0]
    logger.cri('CG Case Name {}'.format(case_id))
    naive_aa_path = os.path.join(configs.save_folder, case_id + '_naive_aa')
    logger.cri('Generate Naive All-Atoms Structure From CG Toplogy Structure')
    genCGReference(configs.cg_top, naive_aa_path)
    logger.cri(f'Saving Naive All-Atoms Structure to {naive_aa_path}')

    # MAPPING STEP
    logger.info('Mapping All-Atoms PDB to CG')
    ref_cg = os.path.join(configs.save_folder, case_id + '_ref_CG.pdb')
    temp_opt = {
        'pdb_aa': configs.ref_pdb,
        'pdb_cg': ref_cg,
        'slice': os.path.join(configs.save_folder, case_id + '-SLICE.json'),
        'mass': os.path.join(configs.save_folder, case_id + '-MASS.json'),
        'map_file': configs.map_file,
        'mapping_mode': 'com'
    }
    logger.cri('Mapping Step Started')
    logger.cri(f'Referecne AA Structures Mapping to {ref_cg}')
    AA2CG(pdb_aa=temp_opt['pdb_aa'], pdb_cg=temp_opt['pdb_cg'], map_file=temp_opt['map_file'], slice=temp_opt['slice'], mass=temp_opt['mass'], mapping_mode=temp_opt['mapping_mode'])
    logger.cri('Mapping Step Finished')

    # CG-Level RMSD STEP
    top1_path = cg_rmsd(cg_traj_path=configs.cg_traj, cg_top_path=configs.cg_top, cg_ref_path=ref_cg, save_folder=configs.save_folder, logger=logger)

    # Energy Cluster STEP
    traj_list_path = evaluate_energy_cluster(cg_traj_path=configs.cg_traj, cg_top_path=configs.cg_top, ref_cg_path=ref_cg, cg_out_path=configs.cg_out,
                            save_folder=configs.save_folder, logger=logger)

    # Average Sturcture STEP
    avg_structure_list_path = cal_average_structure(traj_list_path=traj_list_path, cg_top_path=configs.cg_top, save_folder=configs.save_folder, logger=logger)
    avg_structure_list_path.append(top1_path)
    
    # Remapping STEP
    refined_structure_list_path = remapping(naive_aa_path=naive_aa_path, avg_strucutre_list_path=avg_structure_list_path, save_folder=configs.save_folder, mapping_file=configs.map_file, logger=logger)
    
    # RNA-BRiQ Refinement
    # if configs.BRiQ:
    #     BRiQ_bin_path = os.path.join(configs.BRiQ_path, 'build/bin')
    #     BRiQ_data_path = os.path.join(configs.BRiQ_path, 'BRiQ_data')

    #     try:
    #         os.system(f'export PATH=$PATH:{BRiQ_bin_path}')
    #     except:
    #         logger.error(f'RNA-BRiQ Path Error, Current Using {BRiQ_bin_path}')

    #     try:
    #         os.system(f'export BRiQ_DATAPATH={BRiQ_data_path}')
    #     except:
    #         logger.error(f'RNA-BRiQ Data Path Error, Current Using {BRiQ_data_path}')

    #     if configs.BRiQ_file == 'None':
    #         logger.cri('RNA-BRiQ input file not found, Using Automatic Generated File')
            
    
    # AA-Level RMSD STEP
    aa_rmsd(refined_structure_list=refined_structure_list_path, ref_aa_path=configs.ref_pdb, logger=logger)
    logger.cri('Evaluation Process Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', type=str, default='test.dcd')
    parser.add_argument('--out', type=str, default='test.out')
    parser.add_argument('--top', type=str, default='./test_cg_ss_out.pdb')
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--BRiQ', default=False)
    parser.add_argument('--BRiQ_path', default='/home/RNA-BRiQ')
    parser.add_argument('--BRiQ_file')
    parser.add_argument('--ref_pdb', type=str, default='ref_test.pdb')
    parser.add_argument('--map_file', type=str, default='/home/MD/RNA/params/mapping_rna.yml')
    parser.add_argument('--save_folder', type=str)

    opt = parser.parse_args()
    

    if opt.save_folder == None:
        case_name = os.path.basename(opt.traj).split('.')[0]
        save_folder = os.path.join(os.path.dirname(os.path.abspath(opt.traj)), case_name+'_evaluation')
    else:
        case_name = os.path.dirname(os.path.join(opt.save_folder, 'test'))
        save_folder = os.path.abspath(opt.save_folder)

    if not os.path.isdir(opt.save_folder):
        os.mkdir(opt.save_folder)
    
    print(f'Save Folder {save_folder}')

    if opt.input_folder == None:
        configs = {
            'cg_traj': opt.traj,
            'cg_out': opt.out,
            'cg_top': opt.top,
            'BRiQ_file': opt.BRiQ_file,
            'BRiQ_path': opt.BRiQ_path,
            'BRiQ': opt.BRiQ,
            'ref_pdb': opt.ref_pdb,
            'map_file': opt.map_file,
            'save_folder': opt.save_folder
        }
    else:
        configs = {
            'cg_traj': os.path.join(opt.input_folder, case_name + '.dcd'),
            'cg_out': os.path.join(opt.input_folder, case_name + '.out'),
            'cg_top': os.path.join(opt.input_folder, case_name + '_cg.pdb'),
            'BRiQ_file': opt.BRiQ_file,
            'BRiQ_path': opt.BRiQ_path,
            'BRiQ': opt.BRiQ,
            'ref_pdb': os.path.join(opt.input_folder, 'ref.pdb'),
            'map_file': opt.map_file,
            'save_folder': opt.save_folder
        }

    dot_configs = DotDict(configs)
    logger_path = os.path.join(opt.save_folder, f'{case_name}-evaluation.log')
    logger = Logger(logger_path)
    logger.info(configs)

    main(dot_configs, logger)