from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
import numpy as np


def standardize_rep(mol_rep):
    if isinstance(mol_rep, str):
        mol_rep = Chem.MolFromSmiles(mol_rep)
    elif isinstance(mol_rep, Chem.Mol):
        mol_rep = mol_rep
    else:
        raise TypeError('Invalid Mol representation')

    return mol_rep


def load_smiles_df(filename, column_names=None, delim='\t'):
    """
    loads smiles data into pandas dataframe

    :param filename: file to load smiles data from.
    :param column_names: columns to load from file. defaults to all
    :param delim: how file is delimited

    :return df: dataframe of smiles
    """
    if column_names:
        df = pd.read_csv(filename, sep=delim, header=0, usecols=column_names)
    else:
        df = pd.read_csv(filename, sep=delim, header=0,low_memory=False)
    return df

def generate_adjacency_mat(mol_rep):
    """
    generate adjacency matrix for given SMILES

    :param mol_rep: Molecule representation of molecule to generate adjacency matrix of

    :return adj_mat: adjacency matrix of input smile
    """

    m = standardize_rep(mol_rep)
    if not m:
        return None
    adj_mat = rdmolops.GetAdjacencyMatrix(m)
    return adj_mat


def generate_edge_list(mol_rep):
    """
    generate the edge list of a given molecule
    :param mol_rep: Molecule representation
    :return edge_list: edge list of smile string
    """
    mol = standardize_rep(mol_rep)
    if not mol:
        return None
    num_bonds = len(mol.GetBonds())
    edge_list = np.zeros((2, num_bonds * 2), np.int64)
    for idx, bond in enumerate(mol.GetBonds()):
        bond_idx1 = bond.GetBeginAtomIdx()
        bond_idx2 = bond.GetEndAtomIdx()
        edge_list[0, idx * 2] = bond_idx1
        edge_list[1, idx * 2] = bond_idx2
        edge_list[0, idx * 2 + 1] = bond_idx2
        edge_list[1, idx * 2 + 1] = bond_idx1

    return edge_list


def generate_atom_feature_matrix(mol_rep, feat_func, vec_size=None):
    """
    takes a molecule and a featurization function as input. Returns featurizations of all atoms
    in molecule as a matrix. If length of feature vector known can include vec_size to save computation.

    :param mol_rep: Molecule representation
    :param feat_func: function used to compute features of an atom
    :param vec_size: if vector size known include to save some computation
    :return feat_mat: feature matrix of atoms
    """
    m = standardize_rep(mol_rep)
    if not m:
        return None
    atoms = m.GetAtoms()
    if vec_size:
        feat_mat = np.zeros((len(atoms), vec_size), np.float32)
    else:
        feat_mat = np.zeros((len(atoms), len(feat_func(atoms[0], m))), np.float32)

    for idx, a in enumerate(atoms):
        feat_mat[idx] = feat_func(a, m)

    return feat_mat


def generate_bond_adj_feat_mat(mol, feat_func, vec_size=None):
    mol = standardize_rep(mol)
    if not mol:
        return None
    atoms = mol.GetAtoms()
    if not vec_size:
        vec_size = len(feat_func(mol.GetBonds()[0]))
    num_atoms = len(atoms)
    feat_mat = np.zeros((vec_size, num_atoms, num_atoms), np.float32)
    for b in mol.GetBonds():
        idx1 = b.GetBeginAtomIdx()
        idx2 = b.GetEndAtomIdx()
        edge_vec = feat_func(b)
        feat_mat[:, idx1, idx2] = edge_vec
        feat_mat[:, idx2, idx1] = edge_vec
    return feat_mat


def build_bond_tensor(mol, feat_func, vec_size=None):
    mol = standardize_rep(mol)
    if not mol:
        return None
    if not vec_size:
        vec_size = len(feat_func(mol.GetBonds()[0]))
    num_bonds = len(mol.GetBonds())
    edge_attr = np.zeros((num_bonds * 2, vec_size),np.float32)
    for idx, bond in enumerate(mol.GetBonds()):
        bond_idx1 = bond.GetBeginAtomIdx()
        bond_idx2 = bond.GetEndAtomIdx()
        edge_attr[idx * 2, :] = feat_func(bond)
        edge_attr[idx * 2 + 1, :] = edge_attr[idx * 2, :]
    return edge_attr


def build_edge_list_and_bond_tensor(mol, feat_func, vec_size=None):
    mol = standardize_rep(mol)
    if not mol:
        return None
    if not vec_size:
        vec_size = len(feat_func(mol.GetBonds()[0]))

    num_bonds = len(mol.GetBonds())
    edge_attr = np.zeros((num_bonds * 2, vec_size), np.float32)
    edge_list = np.zeros((2, num_bonds * 2), np.int64)
    for idx, bond in enumerate(mol.GetBonds()):
        bond_idx1 = bond.GetBeginAtomIdx()
        bond_idx2 = bond.GetEndAtomIdx()
        edge_attr[idx * 2, :] = feat_func(bond)
        edge_attr[idx * 2 + 1, :] = edge_attr[idx * 2, :]
        edge_list[0, idx * 2] = bond_idx1
        edge_list[1, idx * 2] = bond_idx2
        edge_list[0, idx * 2 + 1] = bond_idx2
        edge_list[1, idx * 2 + 1] = bond_idx1
    return edge_attr, edge_list


def get_bond_vec(bond):
    bond_map = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}
    vec_size = 4
    bond_vec = np.zeros((vec_size,), np.float32)
    bond_vec[bond_map[str(bond.GetBondType())]] = 1.
    return bond_vec

def get_basic_atom_vec(atom, mol):
    atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 50: 9, 53: 10, 5 : 11, 29: 12, 30:13,
                27:14, 25: 15, 33: 16, 13: 17, 28: 18, 34:19, 14:20, 23:21, 40:22, 3:23, 51:24, 26:25, 46:26,80:27,
                83:28, 11:29, 20:30, 22:31,67:32,32:33,78:34,44:35,45:36, 24:37, 31:38, 19:39, 47:40, 79:41, 65:42,
                77:43, 52: 44, 12 : 45, 82: 46, 74:47, 55:48, 42:49, 75:50, 92:51, 64:52, 81:53, 89:54}
    hyb_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 13, 5: 13}
    vec_size = len(atom_map)
    atom_vec = np.zeros((vec_size,), np.float32)
    atom_vec[atom_map[atom.GetAtomicNum()]] = 1.
    # atom_vec[hyb_map[atom.GetHyb()]] = 1.
    # atom_vec[10] = atom.GetFormalCharge()
    #atom_vec[10] = atom.G()
    # atom_vec[16] = 1. if atom.IsChiral() else 0.
    # atom_vec[17] = 1. if atom.IsInRing() else  0.
    # atom_vec[18] = 1. if atom.IsAromatic() else 0.
    # coords = oe_mol.GetCoords(atom)
    # atom_vec[19] = coords[0]
    return atom_vec

def get_pos_atom_vec(atom, mol):

    atom_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 50: 9, 53: 10, 5: 11}
    hyb_map = {0: 10, 1: 11, 2: 12, 3: 13, 4: 13, 5: 13}
    vec_size = 16
    atom_vec = np.zeros((vec_size,), np.float32)
    atom_vec[atom_map[atom.GetAtomicNum()]] = 1.
    pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
    atom_vec[12] = pos.x
    atom_vec[13] = pos.y
    atom_vec[14] = pos.z
    atom_vec[15] = float(atom.GetProp('_TriposPartialCharge'))
    return atom_vec

def construct_mol_from_graph(adj,feat,edge_feat):

    #TODO: test thouroughly

    flip_atom_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F', 5: 'P', 6: 'S', 7: 'Cl', 8: 'Br', 9: 'I'}
    bond_map = {0 : Chem.rdchem.BondType.SINGLE, 1 : Chem.rdchem.BondType.DOUBLE,
                2 : Chem.rdchem.BondType.TRIPLE, 3 : Chem.rdchem.BondType.AROMATIC}
    e_mol = Chem.EditableMol(Chem.Mol())
    atom_type_idcs = np.argwhere(feat[:,:10])
    for idx in atom_type_idcs[:,1]:
        atomtype = flip_atom_map[idx]
        e_at = Chem.Atom(atomtype)
        e_mol.AddAtom(e_at)
    bond_type_idcs = np.argwhere(edge_feat)
    prev = []
    for idx in bond_type_idcs:
        if [int(idx[2]),int(idx[1])] in prev:
            break
        bondtype = bond_map[idx[0]]
        e_mol.AddBond(int(idx[1]),int(idx[2]),order=bondtype)
        prev.append([int(idx[1]),int(idx[2])])
    return e_mol.GetMol()


def get_mol_from_mol2(molfile, removeH=True):
    return Chem.MolFromMol2File(molfile, sanitize=False, removeHs=removeH)
