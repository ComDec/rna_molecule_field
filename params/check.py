import os
import xml.etree.ElementTree as ET
import numpy as np

path = '/home/MD/RNA'

mapping = {
    'BP': 'Q0',
    'B1': 'SN0',
    'B2': 'SNda',
    'SC1-A': 'TN0',
    'SC2-A': 'TNa',
    'SCD-A': 'TA3',
    'SCA-A': 'TA2',
    'SC1-G': 'TN0',
    'SC2-G': 'TNa',
    'SCD-G': 'TG3',
    'SCA-G': 'TG2',
    'SC1-C': 'TN0',
    'SCA-C': 'TY3',
    'SCD-C': 'TY2',
    'SC1-U': 'TN0',
    'SCA-U': 'TT3',
    'SCD-U': 'TT2',
}

d = np.loadtxt('./LJ.itp', dtype=str)

tree = ET.parse('./RNA-martini.xml')
root = tree.getroot()
sub = root.find('LennardJonesForce')
sub1 = sub.findall('NBFixPair')

for i in range(len(sub1)):
    t_sub = sub1[i]
    class1 = t_sub.attrib['class1']
    class2 = t_sub.attrib['class2']
    sigma = .0
    eps = .0
    for j in range(d.shape[0]):
        lines = d[j]
        if mapping[class1] == lines[0] and mapping[class1] == lines[1]:
            sigma = lines[-2]
            eps = lines[-1]
        if mapping[class1] == lines[1] and mapping[class1] == lines[0]:
            sigma = lines[-2]
            eps = lines[-1]
    t_sub.attrib['sigma'] = str(np.around(float(sigma), 2))
    t_sub.attrib['epsilon'] = str(np.around(float(eps), 2))

tree.write('RNA-test.xml')