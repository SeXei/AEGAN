import os
import json
import numpy as np
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import torch.nn.functional as F
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
cmd = 'psiblast -query _temp.fasta -db {} -num_iterations 3 -evalue 0.001 -num_alignments 1500 -out_ascii_pssm _temp.pssm -out result.txt -comp_based_stats 1 -num_threads 40'
one_hot = [0 for _ in range(21)]
transRes = {
    'ASP': 'D',
    'PRO': 'P',
    'ARG': 'R',
    'ILE': 'I',
    'LEU': 'L',
    'GLU': 'E',
    'ALA': 'A',
    'VAL': 'V',
    'TYR': 'Y',
    'ASN': 'N',
    'HIS': 'H',
    'PHE': 'F',
    'THR': 'T',
    'GLN': 'Q',
    'GLY': 'G',
    'MET': 'M',
    'LYS': 'K',
    'SER': 'S',
    'CYS': 'C',
    'TRP': 'W',
}
Atchley = {
    'A': [-0.591, -1.302, -0.733, 1.57, -0.146],
    'C': [-1.343, 0.465, -0.862, -1.02, -0.255],
    'D': [1.05, 0.302, -3.656, -0.259, -3.242],
    'E': [1.357, -1.453, 1.477, 0.113, -0.837],
    'F': [-1.006, -0.59, 1.891, -0.397, 0.412],
    'G': [-0.384, 1.652, 1.33, 1.045, 2.064],
    'H': [0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
    'K': [1.831, -0.561, 0.533, -0.277, 1.648],
    'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
    'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
    'N': [0.945, 0.828, 1.299, -0.169, 0.933],
    'P': [0.189, 2.081, -1.628, 0.421, -1.392],
    'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [1.538, -0.055, 1.502, 0.44, 2.897],
    'S': [-0.228, 1.399, -4.76, 0.67, -2.647],
    'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
    'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
    'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
    'Y': [0.26, 0.83, 3.097, -0.838, 1.512]
}


def generatePssm(Sequence):
    #     print("start to generate pssminfo")
    Pssm = list()
    # 生成并获取Pssm
    with open('_temp.fasta', 'w') as f:
        f.write('>temp\n' + Sequence)
    os.system(cmd)
    with open('_temp.pssm', 'r') as f:
        data = f.readlines()
    for line in data:
        line = [item for item in line.strip().split(' ') if item not in [' ', '']]
        if len(line) != 0 and line[0].isdigit():
            # Pssm.append(line[2: 22])
            pssm = list(map(float, line[2: 22]))
            try:
                pssm = pssm + Atchley[line[1]]
            except:
                pssm = pssm + [0, 0, 0, 0, 0]
            Pssm.append(pssm)
    #     print("pssminfo has generated!")
    os.remove('./_temp.fasta')
    os.remove('./_temp.pssm')
    #     print("pssminfo has load!")
    return Pssm


def NeighborNodeGenerator(cblist, Nodes, index):
    # 计算最近
    dist = []
    for i in index:
        dist.append(struc.distance(cblist[i], cblist))
    dist = torch.tensor(np.array(dist))
    index = torch.topk(dist, Nodes, dim=-1, largest=False)[1]
    return index


def err_handle(atom, dsc):
    for i in range(3, 7):
        if atom[i] == 0:
            atom[i] = atom[i - 1]
    with open("./err_atom", 'a+') as f:
        f.write(dsc + '\n')
    return atom


def get_atom(subarray, dscribe):
    res_name = subarray[0].res_name
    try:
        atom = [0, 0, 0, 0, 0, 0, 0]
        atom[0] = subarray[0]
        atom[1] = subarray[1]
        atom[2] = subarray[2]
        if res_name == 'SER':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'OG'][0]
        elif res_name == 'PHE':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD1'][0]
            atom[6] = subarray[subarray.atom_name == 'CZ'][0]
        elif res_name == 'THR':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG2'][0]
            atom[5] = subarray[subarray.atom_name == 'OG1'][0]
        elif res_name == 'LEU':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD1'][0]
            atom[6] = subarray[subarray.atom_name == 'CD2'][0]
        elif res_name == 'ASN':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'OD1'][0]
            atom[6] = subarray[subarray.atom_name == 'ND2'][0]
        elif res_name == 'LYS':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD'][0]
            atom[6] = subarray[subarray.atom_name == 'NZ'][0]
        elif res_name == 'VAL':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG1'][0]
            atom[5] = subarray[subarray.atom_name == 'CG2'][0]
        elif res_name == 'ILE':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG1'][0]
            atom[5] = subarray[subarray.atom_name == 'CG2'][0]
            atom[6] = subarray[subarray.atom_name == 'CD1'][0]
        elif res_name == 'ALA':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
        elif res_name == 'GLU':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'OE1'][0]
            atom[5] = subarray[subarray.atom_name == 'OE2'][0]
        elif res_name == 'ARG':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'NE'][0]
            atom[6] = subarray[subarray.atom_name == 'NH2'][0]
        elif res_name == 'ASP':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'OD1'][0]
            atom[6] = subarray[subarray.atom_name == 'OD2'][0]
        elif res_name == 'PRO':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD'][0]
        elif res_name == 'TYR':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD1'][0]
            atom[6] = subarray[subarray.atom_name == 'OH'][0]
        elif res_name == 'GLN':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CD'][0]
            atom[6] = subarray[subarray.atom_name == 'NE2'][0]
        elif res_name == 'HIS':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'ND1'][0]
            atom[6] = subarray[subarray.atom_name == 'CE1'][0]
        elif res_name == 'TRP':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'CE3'][0]
            atom[6] = subarray[subarray.atom_name == 'CH2'][0]
        elif res_name == 'CYS':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'SG'][0]
        elif res_name == 'MET':
            atom[3] = subarray[subarray.atom_name == 'CB'][0]
            atom[4] = subarray[subarray.atom_name == 'CG'][0]
            atom[5] = subarray[subarray.atom_name == 'SD'][0]
            atom[6] = subarray[subarray.atom_name == 'CE'][0]
    except:
        atom = err_handle(atom, dscribe + f"_{res_name}")
    return atom


def getAtomModel(length, array, desc):
    #     base = array[0].res_id
    ca = array[(array.atom_name == 'CA') & (array.hetero == False)]
    AM = []
    for i in range(length):
        subarray = array[(array.res_id == ca[i].res_id) & (array.hetero == False)]
        atom = get_atom(subarray, desc + f'_{ca[i].res_id}')
        atomModel = [0, 0, 0]
        if subarray[0].res_name == 'SER':
            atomModel = atom[4].coord - atom[1].coord
        elif subarray[0].res_name in ['PHE', 'LEU', 'ASN', 'LYS', 'ARG', 'ASP', 'TYR', 'GLN', 'TRP']:
            atomModel = 1 / 2 * (atom[6].coord + atom[5].coord) - atom[1].coord
        elif subarray[0].res_name in ['THR', 'VAL', 'GLU']:
            atomModel = 1 / 2 * (atom[5].coord + atom[4].coord) - atom[1].coord
        elif subarray[0].res_name == 'ILE':
            atomModel = 1 / 4 * (atom[5].coord + atom[4].coord) + 1 / 2 * atom[6].coord - atom[1].coord
        elif subarray[0].res_name == 'ALA':
            atomModel = atom[3].coord - atom[1].coord
        elif subarray[0].res_name == 'PRO':
            atomModel = atom[5].coord + atom[3].coord - atom[4].coord - atom[1].coord
        elif subarray[0].res_name in ['HIS', 'MET']:
            atomModel = 1 / 4 * (atom[4].coord + atom[6].coord) + 1 / 2 * atom[5].coord - atom[1].coord
        elif subarray[0].res_name == 'CYS':
            atomModel = atom[4].coord - atom[1].coord
        if subarray[0].res_name != 'GLY':
            angle = struc.dihedral(atom[4], atom[3], atom[1], atom[3])
            atomModel = np.append([np.sin(angle), np.cos(angle)], atomModel)
        else:
            atomModel = np.append([0, 0], atomModel)
        AM.append(atomModel)
    return torch.tensor(np.array(AM))


def get_edge(cb, neighbor):
    cb = torch.tensor(cb)
    node_cb = cb[neighbor]  # [b, s, d]
    b, s, d = node_cb.shape
    x = node_cb.unsqueeze(2).repeat(1, 1, s, 1).reshape(b, s * s, d)
    y = node_cb.unsqueeze(1).repeat(1, s, 1, 1).reshape(b, s * s, d)
    unit = F.normalize(x - y)
    pdist = torch.nn.PairwiseDistance(p=2)
    dist = pdist(x, y)
    adj = (dist < 8).reshape(b, s, s).float()
    dist = torch.div(dist, 0.5, rounding_mode='floor').long()
    dist[dist > 31] = 31
    dist_code = F.one_hot(dist, num_classes=32)
    return torch.cat([dist_code, unit], dim=-1), adj


def _site_handle_(CB, phi_psi_omega, neighbor_info, pssm_info, AM, gap, target, pbdname):
    node_pssm = pssm_info[neighbor_info]
    node_torsion = phi_psi_omega[neighbor_info]
    node_AM = AM[neighbor_info]
    node_fea = torch.cat([node_pssm, node_torsion, node_AM], dim=-1)
    edge, adj = get_edge(CB.coord, neighbor_info)
    node_fea[torch.isnan(node_fea)] = 0
    node_fea[torch.isinf(node_fea)] = 0
    edge[torch.isinf(edge)] = 0
    edge[torch.isnan(edge)] = 0
    for i in range(len(neighbor_info)):
        if i < gap:
            label = 0
        else:
            label = 1
        torch.save([node_fea[i], edge[i], adj[i], label], os.path.join(target, pbdname + '_{}'.format(i)))


def sofmax(logits):
    e_x = np.exp(logits)
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return probs


def guss_like_generator(mean, x, peak):
    mean = np.array(mean)[:, None]
    std = len(x)
    return (np.exp(-np.power((x - mean), 2) / (2 * std)) / (peak * 1.1)).sum(axis=0)


def choose_by_prob(prob, active):
    prob = sofmax(prob)
    act_nums = len(active)
    active = np.array(active)
    sample_num = act_nums * 4 if act_nums * 4 < len(prob) else len(prob)
    choose = np.random.choice(list(range(len(prob))), sample_num, p=prob)
    deredundance = list(set(choose) - set(active))
    return deredundance


def sample(active_index, length, is_all=False):
    peak = len(active_index)
    # 进行挑选
    if not is_all:
        chosen_index = choose_by_prob(
            guss_like_generator(active_index, np.arange(length), peak),
            active_index
        )
    else:
        chosen_index = list(set(range(length)) - set(active_index))
    return chosen_index


def constructData(chain, pdbname, activeindex, target, desc, is_all):
    ca_list = chain[(chain.atom_name == "CA") & (chain.hetero == False)]
    phi, psi, omega = struc.dihedral_backbone(chain)
    phi_psi_omega = torch.stack([torch.tensor(phi), torch.tensor(psi), torch.tensor(omega)]).transpose(-1, -2)
    phi_psi_omega = torch.cat([torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1)
    phi_psi_omega = torch.nan_to_num(phi_psi_omega)
    # 生成并获取pssm矩阵
    Sequence = ''.join(list(map(transRes.get, ca_list.res_name.tolist())))
    seq_len = len(Sequence)
    if seq_len < 30:
        return
    Pssm_Atchley = torch.tensor(generatePssm(Sequence))
    indexinfo = sample(activeindex, len(ca_list), is_all)
    AM = getAtomModel(len(Sequence), chain, desc + f'_{pdbname}')
    cb = chain[
        (((chain.res_name == 'GLY') * (chain.atom_name == 'CA')) | (chain.atom_name == 'CB')) & (chain.hetero == False)]
    NeighborInfo = NeighborNodeGenerator(cb, 21, indexinfo + activeindex)
    _site_handle_(cb, phi_psi_omega, NeighborInfo, Pssm_Atchley, AM, len(indexinfo), target, pdbname)
    pass


def mainwork(pdbfilepath, targetpath, activeindexpath, is_all):
    if not os.path.exists(targetpath):
        os.makedirs(targetpath)
    desc = pdbfilepath.split("/")[-2]
    with open(activeindexpath, 'r') as f:
        siteindex = json.load(f)
    for pdbfile in tqdm(siteindex.keys()):
        pdb_file = pdb.PDBFile.read(os.path.join(pdbfilepath, pdbfile))
        arry = pdb_file.get_structure(model=1)
        activeindex = siteindex[pdbfile]
        constructData(arry, pdbfile.split(".")[0], activeindex, targetpath, desc, is_all)
    pass


def parse_args():
    parser.add_argument("-s", "--setname", type=str, required=True, help="data set name to be processed")
    parser.add_argument("-t", "--target", default=None, type=str, required=False, help="processed dataset path")
    parser.add_argument("-d", "--database", type=str, required=True, help="Sequence alignment database path for psi-blast")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source_map = {
        "uni14230": ("../database/uni14230pdbfile/", "../database/uni14230.json", "../database/trainset/"),
        "uni3175": ("../database/uni3175pdbfile/", "../database/uni3175.json", "../database/validateset/"),
        "EF_family": ("../database/BenchmarkDataset/EF_family/", "../database/BenchmarkDataset/EF_family_uni_index.json", "../database/BenchmarkDataset_Processed/EF_family"),
        "EF_fold": ("../database/BenchmarkDataset/EF_fold/", "../database/BenchmarkDataset/EF_fold_uni_index.json", "../database/BenchmarkDataset_Processed/EF_fold"),
        "EF_superfamily": (
        "../database/BenchmarkDataset/EF_superfamily/","../database/BenchmarkDataset/EF_superfamily_uni_index.json", "../database/BenchmarkDataset_Processed/EF_superfamily"),
        "HA_superfamily": (
            "../database/BenchmarkDataset/EF_superfamily/",
            "../database/BenchmarkDataset/HA_superfamily_uni_index.json", "../database/BenchmarkDataset_Processed/HA_superfamily"),
        "NN": ("../database/BenchmarkDataset/NN/",
               "../database/BenchmarkDataset/NN_uni_index.json", "../database/BenchmarkDataset_Processed/NN"),
        "PC": ("../database/BenchmarkDataset/EF_fold/",
               "../database/BenchmarkDataset/PC_uni_index.json", "../database/BenchmarkDataset_Processed/PC"),

    }
    source_path, index_path, target_path = source_map.get(args.setname)
    cmd = cmd.format(args.database)
    if args.database is None:
        args.database = target_path
    mainwork(source_path, args.database, index_path, False)
