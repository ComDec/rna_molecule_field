from sklearn.decomposition import PCA
import numpy as np
import mdtraj as md
traj_aa = md.load('./data/AAAAAAGU/mol_cg.dcd', top='./data/AAAAAAGU/mol_cg_fixed.pdb')[-2000:]
traj_cg = md.load('./run_AAAAAAGU.dcd', top='./data/AAAAAAGU/mol_cg_fixed.pdb')[-2000:]
traj_cg_trained_1 = md.load('run_AAAAAAGU_train.dcd', top='./data/AAAAAAGU/mol_cg_fixed.pdb')[-2000:]
traj_cg_trained_2 = md.load('run_AAAAAAGU_train_1.dcd', top='./data/AAAAAAGU/mol_cg_fixed.pdb')[-2000:]
traj_cg_trained_3 = md.load('run_AAAAAAGU_train_2.dcd', top='./data/AAAAAAGU/mol_cg_fixed.pdb')[-2000:]

traj_aa = traj_aa.superpose(traj_aa[0])
traj_cg = traj_cg.superpose(traj_aa[0])
traj_cg_trained_1 = traj_cg_trained_1.superpose(traj_aa[0])
traj_cg_trained_2 = traj_cg_trained_2.superpose(traj_aa[0])
traj_cg_trained_3 = traj_cg_trained_3.superpose(traj_aa[0])

pcaer_aa = PCA(n_components=2)
feature_aa = pcaer_aa.fit_transform(traj_aa.xyz.reshape((traj_aa.n_frames, -1)))
com_aa = pcaer_aa.components_

feature_cg = np.dot(traj_cg.xyz.reshape([traj_cg.n_frames, -1]), com_aa.T)
feature_cg_trained_1 = np.dot(traj_cg_trained_1.xyz.reshape([traj_cg_trained_1.n_frames, -1]), com_aa.T)
feature_cg_trained_2 = np.dot(traj_cg_trained_2.xyz.reshape([traj_cg_trained_2.n_frames, -1]), com_aa.T)
feature_cg_trained_3 = np.dot(traj_cg_trained_3.xyz.reshape([traj_cg_trained_3.n_frames, -1]), com_aa.T)

import matplotlib.pyplot as plt

plt.scatter(feature_aa[:,0], feature_aa[:,1], c="red", alpha=0.2, label="AA")
plt.scatter(feature_cg[:,0], feature_cg[:,1], c="blue", alpha=0.2, label="CG")
plt.scatter(feature_cg_trained_1[:,0], feature_cg_trained_1[:,1], c="green", alpha=0.2, label="CG-train-A")
plt.scatter(feature_cg_trained_2[:,0], feature_cg_trained_2[:,1], c="black", alpha=0.2, label="CG-train-B")
plt.scatter(feature_cg_trained_3[:,0], feature_cg_trained_3[:,1], c="pink", alpha=0.2, label="CG-train-C")

plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.legend()
plt.show()
plt.savefig("pca.png")