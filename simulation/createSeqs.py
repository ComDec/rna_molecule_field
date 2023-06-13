import sys, os
import numpy as np

bp = ["A", "C", "U", "G"]
bplist = []
for ii in bp:
    for jj in bp:
        for kk in bp:
            for ll in bp:
                for mm in bp:
                    for nn in bp:
                        for oo in bp:
                            for pp in bp:
                                bplist.append((ii, jj, kk, ll, mm, nn, oo, pp))

for seq in bplist:
    if np.random.random() > 0.05:
        continue
    txt = "".join(seq)
    os.mkdir("data/"+txt)
    with open(f"data/{txt}/seq.inp", "w") as f:
        f.write("source leaprc.RNA.OL3\n")
        f.write("seq = sequence { %s5 %s %s %s %s %s %s %s3 }\n"%(seq[0], seq[1], seq[2], seq[3], seq[4], seq[5], seq[6], seq[7]))
        f.write(f"savepdb seq seq.pdb\n")
        f.write("quit\n")

