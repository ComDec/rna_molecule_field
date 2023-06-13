import os

root_path = "/data/data/RNA_3D/test"
cg_root = "/data/data/RNA_3D_CG/test"

if not os.path.exists(cg_root):
    os.mkdir(cg_root)

for all_list in os.listdir(root_path):
    id = all_list.split('.')[0]
    cg_id = id + '_cg.pdb'
    aa_path = os.path.join(root_path, all_list)
    cg_path = os.path.join(cg_root, cg_id)
    cmd = f"/root/anaconda3/envs/torch/bin/python /data/RNA/utils/AA2CG_frame.py --pdb_aa {aa_path} --pdb_cg {cg_path} --map_file /data/RNA/utils/mapping_unirna_v0.yml"
    os.system(cmd)