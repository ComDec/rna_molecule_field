from RNApdbeeHtml import HtmlParser
from RNApdbee import RNApdbee3D


def cmap_extractor(pdb_path, return_type='pairmap'):
    html = RNApdbee3D.execute(file_path=pdb_path)
    result = HtmlParser.parse3d(html)
    bpseq = result['BPSEQ']

    return bpseq


if __name__ == '__main__':
    bpseq = cmap_extractor('/home/MD/RNA/rna_puzzle_dataset/pdb/PZ21.pdb')
    print()