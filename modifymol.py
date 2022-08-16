from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
import numpy as np
import moldata

def check_valid_mol(mol):
    a = Chem.MolToSmiles(mol)
    if a:
        return True
    return False

def mutate_mol(mol, min_mutations=1):
    mol = moldata.standardize_rep(mol)
    if not mol:
        return None
    emol = Chem.EditableMol(mol)
    a_list = mol.GetAtoms()
    b_list = mol.GetBonds()
    num_atoms = len(a_list)
    num_bonds = len(b_list)
    for _ in range(min_mutations):
        rand_a = np.random.randint(0, num_atoms)
        emol.RemoveAtom(rand_a)

    new_mol = emol.GetMol()
    #if check_valid_mol(new_mol):
    #    mutate_mol(new_mol)
    return new_mol


