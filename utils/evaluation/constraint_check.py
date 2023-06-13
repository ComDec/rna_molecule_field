import os
from datetime import datetime
import codecs
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def parse_dssr_results(pdb_path, auto_delete=False, with_time=False):
    """
        Input: DSSR commands
        Output: combined DSSR Results
    """
    abs_path = os.path.abspath(pdb_path)
    dirname = os.path.dirname(abs_path)
    basename = os.path.basename(abs_path)
    if with_time:
        file_id = basename.split('.')[0] + '_' + datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        file_id = basename.split('.')[0]
    save_folder = file_id + '_dssr_results'
    save_file = file_id + '.json'
    save_folder_path = os.path.join(dirname, save_folder)
    save_file_path = os.path.join(save_folder_path, save_file)

    os.mkdir(save_folder_path)
    cmd = f'/usr/bin/x3dna-dssr -i={pdb_path} --json -o={save_file_path}'
    os.system(cmd)
    
    cmd = f'mv {dirname}/dssr-* {save_folder_path}'
    os.system(cmd)

    with codecs.open(save_file_path, 'r', 'utf-8') as f:
            dssr_ensembles = json.load(f)

    if auto_delete:
        cmd = f'rm -rf {save_folder_path}'
        os.system(cmd)

    return dssr_ensembles

def convert_dotbracket_to_matrix(s):
    m = np.zeros([len(s), len(s)])
    for char_set in [['(', ')'], ['[', ']'], ['{', '}'], ['<', '>']]:
        bp1 = []
        bp2 = []
        for i, char in enumerate(s):
            if char == char_set[0]:
                bp1.append(i)
            if char == char_set[1]:
                bp2.append(i)
        for i in list(reversed(bp1)):
            for j in bp2:
                if j > i:
                    m[i, j] = 1.0
                    bp2.remove(j)
                    break
    return m

def constraint_evalutaion(predict_pdb, label_pdb):
    predict_json = parse_dssr_results(predict_pdb, auto_delete=True)
    label_json = parse_dssr_results(label_pdb, auto_delete=True)
    predict_matrix = convert_dotbracket_to_matrix(predict_json['dbn']['all_chains']['sstr'])
    tril_predict_matirx = predict_matrix.reshape(-1)
    label_matrix = convert_dotbracket_to_matrix(label_json['dbn']['all_chains']['sstr'])
    tril_label_matrix = label_matrix.reshape(-1)
    return {
        'acc': accuracy_score(tril_predict_matirx, tril_label_matrix),
        'f1': f1_score(tril_predict_matirx, tril_label_matrix, average='macro'),
        'precision': precision_score(tril_predict_matirx, tril_label_matrix, average='macro'),
        'recall': recall_score(tril_predict_matirx, tril_label_matrix, average='macro')
    }


if __name__ == '__main__':
    os.chdir('/home/MD/RNA/prediction')
    results = parse_dssr_results(pdb_path='/home/PMFs/all_relaxed/7ABZ_1_4.pdb', auto_delete=False)
    # metrics = constraint_evalutaion('/home/MD/RNA/prediction/PZ21_Rhofold/refined_structures/CG_BEST_refined.pdb', '/home/MD/RNA/prediction/PZ21_label.pdb')
    print(results)