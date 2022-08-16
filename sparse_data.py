import torch

class SparseNetData:
    """
    represents a sparse representation of graph data that allows for easy access to attributes as well as efficient batching
    of multiple graphs.
    """
    def __init__(self, node_attr=None, edge_attr=None, edge_list=None, y=None, batching=None, get_mask=False):

        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.edge_list = edge_list
        self.batching = batching
        self.node_mask = None
        self.edge_mask = None
        if get_mask:
            self.node_mask = self.get_node_mask()
            self.edge_mask = self.get_edge_mask()

        if self.batching is None:
            self.batching = torch.ones([1, self.num_nodes], dtype=torch.float32)

        self.num_graphs = self.batching.size(0)
        self.y = y

    @classmethod
    def from_dict(cls, dictionary):

        data = cls()
        for key, item in dictionary.items():
            data[key] = item

        return data

    def __getitem__(self, key):

        return getattr(self, key, None)

    def __setitem__(self, key, value):

        setattr(self, key, value)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    @property
    def num_edge_features(self):
        if self.edge_attr is None:
            return 0
        return 1 if self.edge_attr.dim() == 1 else self.edge_attr.size(1)

    @property
    def num_node_features(self):
        if self.node_attr is None:
            return 0
        return 1 if self.node_attr.dim() == 1 else self.node_attr.size(1)

    @property
    def num_edges(self):
        # assumes directed
        return self.edge_list.size(1)

    @property
    def num_nodes(self):
        # assumes directed
        return self.node_attr.size(0)

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            if torch.is_tensor(item):
                self[key] = func(item)
        return self

    def to(self, device, *keys):

        return self.apply(lambda x: x.to(device))

    def get_node_mask(self):
        if self.node_attr is None:
            return None
        else:
            ones = torch.ones(self.num_edges)
            mask = torch.sparse.FloatTensor(self.edge_list, ones, torch.Size([self.num_nodes, self.num_nodes]))
            return mask

    def get_edge_mask(self):
        if self.edge_attr is None:
            return None
        else:
            ones = torch.ones(self.num_edges)
            mask_idx = torch.stack([self.edge_list[1], torch.arange(0, self.num_edges)])
            mask = torch.sparse.FloatTensor(mask_idx, ones, torch.Size([self.num_nodes, self.num_edges]))
            return mask

    def make_edge_mask_dense(self):
        self.edge_mask = self.edge_mask.to_dense()

    def make_node_mask_dense(self):
        self.node_mask = self.node_mask.to_dense()

    def set_masks(self):
        self.node_mask = self.get_node_mask()
        self.edge_mask = self.get_edge_mask()

    def require_grad(self):
        self.node_attr.requires_grad_(True)
        self.edge_attr.requires_grad_(True)
        self.node_mask.requires_grad_(True)
        self.edge_mask.requires_grad_(True)

    def append_(self, other):
        self.edge_list = torch.cat([self.edge_list, other.edge_list + self.node_attr.size(0)], dim=1)
        batching_temp = torch.zeros(self.batching.size(0) + other.batching.size(0), self.batching.size(1) + other.batching.size(1))
        batching_temp[:self.batching.size(0), :self.batching.size(1)] = self.batching
        batching_temp[self.batching.size(0):, self.batching.size(1):] = other.batching
        self.batching = batching_temp
        self.node_attr = torch.cat([self.node_attr, other.node_attr])
        self.edge_attr = torch.cat([self.edge_attr, other.edge_attr])
        self.y = torch.cat([self.y, other.y], dim=0)
        self.node_mask = self.get_node_mask()
        self.edge_mask = self.get_edge_mask()



def collate_sparse(batch,edge_features=False):

    node_counts = [graph.num_nodes for graph in batch]
    total_nodes = sum(node_counts)
    batching = torch.zeros([len(batch),total_nodes])
    edge_lists = [graph.edge_list for graph in batch]
    accum = 0

    for idx, val in enumerate(node_counts):
        batching[idx, accum:accum+val] = 1.
        edge_lists[idx] += accum
        accum += val

    node_stack = torch.cat([graph.node_attr for graph in batch], dim=0)
    edge_lists = torch.cat(edge_lists, dim=1)
    ys = torch.cat([graph.y for graph in batch], dim=0)

    edge_stack = None
    if edge_features:
        edge_stack = torch.cat([graph.edge_attr for graph in batch], dim=0)

    return SparseNetData(node_attr=node_stack, edge_attr=edge_stack, edge_list=edge_lists, y=ys, batching=batching, get_mask=True)