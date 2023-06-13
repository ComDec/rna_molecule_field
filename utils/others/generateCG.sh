for fname in data_v2/*
do
    python AA2CG_frame.py --pdb_aa $fname/mol.pdb --pdb_cg $fname/mol_cg.pdb --slice rna_slice.json --mass rna_mass.json --map_file mapping_rna_test.yml
    python AA2CG_traj.py --traj_aa $fname/mol.dcd --traj_cg $fname/mol_cg.dcd --pdb_aa $fname/mol.pdb --pdb_cg $fname/mol_cg.pdb --slice rna_slice.json --mass rna_mass.json
done