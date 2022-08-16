import torch
from torch_geometric.data import Dataset
from torch.utils.data import Dataset as Tdataset
from sparse_data import SparseNetData
from torch_geometric.data import Data
import os
import numpy as np

class MolGeoDataset(Dataset):

    def __init__(self, root, transform=None, pre_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ordered_files = sorted([x for x in os.listdir(root) if '.pt' in x])
        super(MolGeoDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.ordered_files

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    def process(self):
        pass

    def get(self, idx):
        data = torch.load(os.path.join(self.root, self.processed_file_names[idx]))
        data.x = data.x.float()
        data.y = data.y.unsqueeze(0).float()
        data.edge_attr = data.edge_attr.float()
        return data


class MolBaseDataset(Tdataset):

    def __init__(self, root, transform=None):

        self.ordered_files = sorted([x for x in os.listdir(root) if '.pt' in x])
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.ordered_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) is list:
            idx = idx[0]
        mol_name = os.path.join(self.root, self.ordered_files[idx])
        data = torch.load(mol_name)
        if self.transform:
            data = self.transform(data)
        #abbrev_data = {'atom_feat_mat': data['atom_feat_mat'],
                       #'edge_feat_list_mat': data['edge_feat_list_mat'], 'edge_list':data['edge_list'], 'label':data['label']}
        return data

class MolSparseDataset(Tdataset):

    def __init__(self, root, transform=None):

        self.ordered_files = sorted([x for x in os.listdir(root) if '.pt' in x])
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.ordered_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) is list:
            idx = idx[0]
        mol_name = os.path.join(self.root, self.ordered_files[idx])
        data = torch.load(mol_name)
        if self.transform:
            data = self.transform(data)

        sp_data = SparseNetData(node_attr = data['node_attr'], edge_attr = data['edge_attr'], edge_list = data['edge_list'], y = data['y'], get_mask=True)

        return sp_data

def label_choice_transform(label, data):
    data['y'] = data['y'][label]
    return data