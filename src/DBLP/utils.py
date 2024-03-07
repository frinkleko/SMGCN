import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import HGBDataset, DBLP
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from itertools import permutations
import warnings

warnings.filterwarnings('ignore')
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_HGBD_data(name):
    try:
        data = HGBDataset(root='./data/{}'.format(name),
                          name=name)[0]
    except:
        raise ValueError('Unknown dataset: {}'.format(name))
    return data


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def First_NN_matrix(orig_sim):
    # Create the first nearest neighbor matrix
    s = orig_sim.shape[0]
    np.fill_diagonal(orig_sim, 0)
    initial_rank = np.argmax(orig_sim, axis=1)
    A = csr_matrix((np.ones(s), (np.arange(s), initial_rank)), shape=(s, s))
    A += csr_matrix((np.ones(s), (np.arange(s), np.arange(s))), shape=(s, s))

    # Convert to dense matrix for connected_components
    A = A.toarray()

    B = A + A.T
    B[B > 1] = 1

    np.fill_diagonal(B, 1)
    B = B.astype(bool)
    return B


def get_component(A):
    # Get the connected components of the graph represented by the matrix A
    n_components, labels = connected_components(A, directed=False)
    return labels


def get_similarity(c, u, data):
    # Create a binary indicator matrix u_ from the unique values of u
    u_ = np.eye(len(np.unique(u)))[u]
    # Get the number of clusters from the shape of u_
    num_clust = u_.shape[0]
    # If c is not empty, use it to index u. Otherwise, just use u as it is.
    # In the original MATLAB code, c is used to index u (not u_, the indicator matrix).
    # This is a critical difference between the MATLAB and Python versions.
    if c.size != 0:
        c = u[c.ravel()]
    else:
        c = u
    # Compute the similarity matrix. This is done by matrix multiplication of the transpose of u_,
    # the data, and u_ itself.
    mat = u_.T @ data @ u_
    # Count the number of instances in each cluster and create a matrix cnt_mul by outer product.
    cnt_mul = np.sum(u_, axis=0).reshape(-1, 1)
    cnt_mul = cnt_mul @ cnt_mul.T
    # Normalize
    mat = mat / cnt_mul
    return c, num_clust, mat


def finch(mat):
    from copy import deepcopy
    print("FINCH")
    print("Input matrix shape: ", mat.shape)
    # Compute the first nearest neighbor matrix from the input similarity matrix
    Affinity_ = First_NN_matrix(mat)
    # Find the connected components in the graph represented by the first nearest neighbor matrix
    cluster_indicator = [get_component(Affinity_)]
    # Compute the similarity matrix from the input matrix and the first set of cluster labels
    Y, num_clust, mat = get_similarity(np.array([]), cluster_indicator[0], mat)

    # Store the initial cluster labels

    c_ = Y

    while True:
        # Recompute the first nearest neighbor matrix with the updated similarity matrix
        Affinity_ = First_NN_matrix(mat)
        # Find the connected components again and append the labels to the cluster indicator
        cluster_indicator.append(get_component(Affinity_))
        # Update the cluster labels, the number of clusters, and the similarity matrix
        c_, num_clust_curr, mat = get_similarity(c_, cluster_indicator[-1],
                                                 mat)
        # Append the current number of clusters to the list of number of clusters
        num_clust = np.append(num_clust, num_clust_curr)
        # Append the updated cluster labels to the matrix of cluster labels
        Y = np.column_stack((Y, c_))
        # If the current number of clusters is 1, or if the difference between the current number of clusters
        # and the previous number of clusters is less than 1, then end the loop
        if num_clust_curr == 1 or num_clust[-2] - num_clust_curr < 1:
            # Remove the last entry from the list of number of clusters and the matrix of cluster labels
            num_clust = num_clust[:-1]
            Y = Y[:, :-1]
            # Break the loop
            break
    # Return the final cluster labels, the number of clusters, and the cluster indicators
    return Y, num_clust, cluster_indicator


def T2(edge_index, num_nodes, self_loop=False):
    edge_index = edge_index.to('cpu')
    adj = np.zeros((num_nodes, num_nodes))
    if self_loop:
        adj = adj + np.eye(num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    weight_adj = adj * (np.dot(adj, adj))
    assert adj.shape == weight_adj.shape
    neighbor = First_NN_matrix(weight_adj)
    return neighbor


def agreeT2(T2s, norm=False):
    # T2s: list of T2
    # T2 is nxn 0-1 matrix
    sum_T2 = np.zeros(T2s[0].shape)
    for T2 in T2s:
        assert T2.shape == T2s[0].shape
        sum_T2 += T2

    if norm:
        pass
    else:
        sum_T2[sum_T2 < 2] = 0
        sum_T2[sum_T2 >= 2] = 1

    return sum_T2


def adj2index(adj):
    adj = torch.from_numpy(adj)
    adj = torch.tensor(adj, dtype=torch.float32)
    print('adj shape', adj.size())
    index = adj.nonzero().t()
    # turn into edge_index
    w = adj[index[0], index[1]]
    return index.to(device), w.to(device)


class TwoLayerGCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels=128):
        super(TwoLayerGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class OneLayerGCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels=128, nonlinear=True):
        super(OneLayerGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.nonlinear = nonlinear
        self.conv1 = GCNConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if self.nonlinear:
            x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return x


class WOneLayerGCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels=128, nonlinear=True):
        super(WOneLayerGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.nonlinear = nonlinear
        self.conv1 = GCNConv(in_channels, hidden_channels)

    def forward(self, x, edge_index, w):
        x = self.conv1(x, edge_index, w)
        if self.nonlinear:
            x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return x


class PGNN(torch.nn.Module):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(PGNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.lin = torch.nn.Linear(3 * hidden_channels, num_classes)
        self.conv1 = OneLayerGCN(in_channels, hidden_channels)
        self.conv2 = OneLayerGCN(in_channels, hidden_channels)
        self.conv3 = OneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3):
        x1 = self.conv1(x, edge_index1)
        x2 = self.conv2(x, edge_index2)
        x3 = self.conv3(x, edge_index3)
        max_x = torch.max(x1, torch.max(x2, x3))
        min_x = torch.min(x1, torch.min(x2, x3))
        mean_x = (x1 + x2 + x3) / 3
        x = torch.cat([max_x, min_x, mean_x], dim=1)
        out = self.lin(x)
        out = F.log_softmax(out, dim=1)
        return out


class MIMOGNN1(torch.nn.Module):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MIMOGNN1, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.conv1 = OneLayerGCN(in_channels, hidden_channels)
        self.conv2 = OneLayerGCN(in_channels, hidden_channels)
        self.conv_id = OneLayerGCN(in_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index1, edge_index2, edge_index_id):
        x1 = self.conv1(x, edge_index1)
        x2 = self.conv2(x, edge_index2)
        x_id = self.conv_id(x, edge_index_id)
        x = x1 + x2 + x_id
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MIMOGNN2(torch.nn.Module):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MIMOGNN2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.conv1 = OneLayerGCN(in_channels, hidden_channels)
        self.conv2 = OneLayerGCN(in_channels, hidden_channels)
        self.conv3 = OneLayerGCN(in_channels, hidden_channels)
        self.conv4 = OneLayerGCN(in_channels, hidden_channels)
        self.conv_id = OneLayerGCN(in_channels, hidden_channels)

        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index_id, edge_index1, edge_index2, edge_index3,
                edge_index4):
        x1 = self.conv1(x, edge_index1)
        x2 = self.conv2(x, edge_index2)
        x3 = self.conv3(x, edge_index3)
        x4 = self.conv4(x, edge_index4)
        x_id = self.conv_id(x, edge_index_id)
        x = x1 + x2 + x3 + x4 + x_id
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MIMOGNN3(torch.nn.Module):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MIMOGNN3, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.conv1 = OneLayerGCN(in_channels, hidden_channels)
        self.conv2 = OneLayerGCN(in_channels, hidden_channels)
        self.conv3 = OneLayerGCN(in_channels, hidden_channels)
        self.conv4 = OneLayerGCN(in_channels, hidden_channels)
        self.conv_id = OneLayerGCN(in_channels, hidden_channels)
        self.conv5 = OneLayerGCN(in_channels, hidden_channels)
        self.conv6 = OneLayerGCN(in_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index_id, edge_index1, edge_index2, edge_index3,
                edge_index4, edge_index5, edge_index6):
        x1 = self.conv1(x, edge_index1)
        x2 = self.conv2(x, edge_index2)
        x3 = self.conv3(x, edge_index3)
        x4 = self.conv4(x, edge_index4)
        x_id = self.conv_id(x, edge_index_id)
        x5 = self.conv5(x, edge_index5)
        x6 = self.conv6(x, edge_index6)
        x = x1 + x2 + x3 + x4 + x_id + x5 + x6
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


def index2adj(edge_index, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        adj[edge_index[0][i]][edge_index[1][i]] = 1
    adj = np.dot(adj, adj)
    adj = torch.tensor(adj, dtype=torch.float32)
    # turn into edge_index
    adj = torch.nonzero(adj)
    adj = adj.T

    return adj


from torch.sparse import mm as spmm
from torch.sparse import FloatTensor


def power_sparse(edge_index, num_nodes):
    sp = FloatTensor(edge_index.to('cpu'), torch.ones(edge_index.size(1)),
                     torch.Size([num_nodes, num_nodes]))
    sp = spmm(sp, sp)
    index = sp.to_dense().nonzero().t()
    w = sp.to_dense()[index[0], index[1]]
    return index.to(device), w.to(device)


def mm_sparse(edge_index1, edge_index2, num_nodes):
    sp1 = FloatTensor(edge_index1.to('cpu'), torch.ones(edge_index1.size(1)),
                      torch.Size([num_nodes, num_nodes]))
    sp2 = FloatTensor(edge_index2.to('cpu'), torch.ones(edge_index2.size(1)),
                      torch.Size([num_nodes, num_nodes]))
    sp = spmm(sp1, sp2)
    return sp.to_dense().nonzero().t().to(device)


def mm_sparse_w(edge_index1, edge_index2, num_nodes):
    sp1 = FloatTensor(edge_index1.to('cpu'), torch.ones(edge_index1.size(1)),
                      torch.Size([num_nodes, num_nodes]))
    sp2 = FloatTensor(edge_index2.to('cpu'), torch.ones(edge_index2.size(1)),
                      torch.Size([num_nodes, num_nodes]))
    sp = spmm(sp1, sp2)
    edge_index = sp.to_dense().nonzero().t()
    edge_value = sp.to_dense()[edge_index[0], edge_index[1]]
    return edge_index.to(device), edge_value.to(device)


def mm_sparse_w(edge_index1, edge_index2, w1, w2, num_nodes):
    sp1 = FloatTensor(edge_index1.to('cpu'), w1.to('cpu'),
                      torch.Size([num_nodes, num_nodes]))
    sp2 = FloatTensor(edge_index2.to('cpu'), w2.to('cpu'),
                      torch.Size([num_nodes, num_nodes]))
    sp = spmm(sp1, sp2)
    edge_index = sp.to_dense().nonzero().t()
    edge_value = sp.to_dense()[edge_index[0], edge_index[1]]
    return edge_index.to(device), edge_value.to(device)


def mm_from_edge_index(index1, index2, w1, w2, num_nodes):
    m = torch.zeros(num_nodes, num_nodes).to(device)
    m[index1[0], index1[1]] = w1
    m2 = torch.zeros(num_nodes, num_nodes).to(device)
    m2[index2[0], index2[1]] = w2
    m = m @ m2
    index = m.nonzero().t()
    w = m[index[0], index[1]]
    return index, w


def symm_edge_index(index1, index2, w1, w2, num_nodes):
    m = torch.zeros(num_nodes, num_nodes).to(device)
    m[index1[0], index1[1]] = w1
    m2 = torch.zeros(num_nodes, num_nodes).to(device)
    m2[index2[0], index2[1]] = w2
    m = (m @ m2 + m2 @ m) / 2
    index = m.nonzero().t()
    w = m[index[0], index[1]]
    return index, w


class SMGNN(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 number_views,
                 hidden_channels=128):
        super(SMGNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.conv = TwoLayerGCN(in_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        out = self.lin(x)
        out = F.log_softmax(out, dim=1)
        return out


def build_view(data, view_name, object_name):
    view_edge = data[view_name, 'to', object_name]['edge_index']
    co_view_edge = torch.tensor([], dtype=torch.long)
    unique_nodes, node_indices = torch.unique(view_edge[0],
                                              return_inverse=True)
    papers_per_node = [
        view_edge[1][node_indices == i] for i in range(unique_nodes.size(0))
    ]
    co_view_edge = torch.cat([
        torch.tensor(list(permutations(papers, 2)), dtype=torch.int64).T
        for papers in papers_per_node
    ],
                             dim=1)
    print('view {} has {} edges'.format(view_name, co_view_edge.shape[1]))
    co_view_graph = Data(edge_index=co_view_edge,
                         x=data[object_name]['x'],
                         y=data[object_name]['y'],
                         train_mask=data[object_name]['train_mask'],
                         test_mask=data[object_name]['test_mask'])
    return co_view_graph


def agree_edge(graphs):
    edge_counter = {}
    for graph in tqdm(graphs):
        for edge in graph.edge_index.T:
            edge = tuple(edge.numpy())
            if edge not in edge_counter:
                edge_counter[edge] = 0
            edge_counter[edge] += 1
    agree_edge_index = []
    # if appear more than once, add to agree_edge_index
    for edge, count in edge_counter.items():
        if count > 1:
            agree_edge_index.append(edge)
    agree_edge_index = torch.tensor(agree_edge_index, dtype=torch.int64).T
    return agree_edge_index


class MGNN1(torch.nn.Module):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN1, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.conv1 = WOneLayerGCN(in_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index1, w1):
        x1 = self.conv1(x, edge_index1, w1)
        x = F.relu(x1)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN2(MGNN1):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN2, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv2 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, w1, w2):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x = x1 + x2
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN3(MGNN2):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN3, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv3 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, w1, w2, w3):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x = x1 + x2 + x3
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN4(MGNN3):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN4, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv4 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                w1, w2, w3, w4):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x = x1 + x2 + x3 + x4
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN5(MGNN4):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN5, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv5 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, w1, w2, w3, w4, w5):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x = x1 + x2 + x3 + x4 + x5
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN6(MGNN5):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN6, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv6 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, w1, w2, w3, w4, w5, w6):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN7(MGNN6):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN7, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv7 = WOneLayerGCN(in_channels, hidden_channels)
        self.lin_init = torch.nn.Linear(in_channels, in_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, w1, w2, w3, w4, w5, w6,
                w7):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)

        x = x1 + x2 + x3 + x4 + x5 + x6 + x7
        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN8(MGNN7):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN8, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv8 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8, w1, w2, w3,
                w4, w5, w6, w7, w8):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN9(MGNN8):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN9, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv9 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, w1, w2, w3, w4, w5, w6, w7, w8, w9):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)

        return x


class MGNN10(MGNN9):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN10, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv10 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, w1, w2, w3, w4, w5, w6, w7, w8, w9,
                w10):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)

        return x


class MGNN11(MGNN10):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN11, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv11 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, w1, w2, w3, w4, w5,
                w6, w7, w8, w9, w10, w11):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)

        return x


class MGNN12(MGNN11):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN12, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv12 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12, w1, w2,
                w3, w4, w5, w6, w7, w8, w9, w10, w11, w12):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)

        return x


class MGNN13(MGNN12):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN13, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv13 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
                w12, w13):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN14(MGNN13):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN14, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv14 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, edge_index14, w1, w2, w3, w4, w5, w6, w7, w8, w9,
                w10, w11, w12, w13, w14):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x14 = self.conv14(x, edge_index14, w14)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN15(MGNN14):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN15, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv15 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, edge_index14, edge_index15, w1, w2, w3, w4, w5,
                w6, w7, w8, w9, w10, w11, w12, w13, w14, w15):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x14 = self.conv14(x, edge_index14, w14)
        x15 = self.conv15(x, edge_index15, w15)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN16(MGNN15):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN16, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv16 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, edge_index14, edge_index15, edge_index16, w1, w2,
                w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x14 = self.conv14(x, edge_index14, w14)
        x15 = self.conv15(x, edge_index15, w15)
        x16 = self.conv16(x, edge_index16, w16)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN17(MGNN16):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN17, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv17 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, edge_index14, edge_index15, edge_index16,
                edge_index17, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
                w12, w13, w14, w15, w16, w17):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x14 = self.conv14(x, edge_index14, w14)
        x15 = self.conv15(x, edge_index15, w15)
        x16 = self.conv16(x, edge_index16, w16)
        x17 = self.conv17(x, edge_index17, w17)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN18(MGNN17):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN18, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv18 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, edge_index14, edge_index15, edge_index16,
                edge_index17, edge_index18, w1, w2, w3, w4, w5, w6, w7, w8, w9,
                w10, w11, w12, w13, w14, w15, w16, w17, w18):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x14 = self.conv14(x, edge_index14, w14)
        x15 = self.conv15(x, edge_index15, w15)
        x16 = self.conv16(x, edge_index16, w16)
        x17 = self.conv17(x, edge_index17, w17)
        x18 = self.conv18(x, edge_index18, w18)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN19(MGNN18):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN19, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv19 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, edge_index14, edge_index15, edge_index16,
                edge_index17, edge_index18, edge_index19, w1, w2, w3, w4, w5,
                w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18,
                w19):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x14 = self.conv14(x, edge_index14, w14)
        x15 = self.conv15(x, edge_index15, w15)
        x16 = self.conv16(x, edge_index16, w16)
        x17 = self.conv17(x, edge_index17, w17)
        x18 = self.conv18(x, edge_index18, w18)
        x19 = self.conv19(x, edge_index19, w19)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN20(MGNN19):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN20, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv20 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, edge_index14, edge_index15, edge_index16,
                edge_index17, edge_index18, edge_index19, edge_index20, w1, w2,
                w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                w17, w18, w19, w20):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x14 = self.conv14(x, edge_index14, w14)
        x15 = self.conv15(x, edge_index15, w15)
        x16 = self.conv16(x, edge_index16, w16)
        x17 = self.conv17(x, edge_index17, w17)
        x18 = self.conv18(x, edge_index18, w18)
        x19 = self.conv19(x, edge_index19, w19)
        x20 = self.conv20(x, edge_index20, w20)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x


class MGNN21(MGNN20):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN21, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv21 = WOneLayerGCN(in_channels, hidden_channels)

    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, edge_index14, edge_index15, edge_index16,
                edge_index17, edge_index18, edge_index19, edge_index20,
                edge_index21, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
                w12, w13, w14, w15, w16, w17, w18, w19, w20, w21):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x14 = self.conv14(x, edge_index14, w14)
        x15 = self.conv15(x, edge_index15, w15)
        x16 = self.conv16(x, edge_index16, w16)
        x17 = self.conv17(x, edge_index17, w17)
        x18 = self.conv18(x, edge_index18, w18)
        x19 = self.conv19(x, edge_index19, w19)
        x20 = self.conv20(x, edge_index20, w20)
        x21 = self.conv21(x, edge_index21, w21)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x

class MGNN22(MGNN21):

    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN22, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv22 = WOneLayerGCN(in_channels, hidden_channels)
    def forward(self, x, edge_index1, edge_index2, edge_index3, edge_index4,
                edge_index5, edge_index6, edge_index7, edge_index8,
                edge_index9, edge_index10, edge_index11, edge_index12,
                edge_index13, edge_index14, edge_index15, edge_index16,
                edge_index17, edge_index18, edge_index19, edge_index20,
                edge_index21,edge_index22, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
                w12, w13, w14, w15, w16, w17, w18, w19, w20, w21,w22):
        x1 = self.conv1(x, edge_index1, w1)
        x2 = self.conv2(x, edge_index2, w2)
        x3 = self.conv3(x, edge_index3, w3)
        x4 = self.conv4(x, edge_index4, w4)
        x5 = self.conv5(x, edge_index5, w5)
        x6 = self.conv6(x, edge_index6, w6)
        x7 = self.conv7(x, edge_index7, w7)
        x8 = self.conv8(x, edge_index8, w8)
        x9 = self.conv9(x, edge_index9, w9)
        x10 = self.conv10(x, edge_index10, w10)
        x11 = self.conv11(x, edge_index11, w11)
        x12 = self.conv12(x, edge_index12, w12)
        x13 = self.conv13(x, edge_index13, w13)
        x14 = self.conv14(x, edge_index14, w14)
        x15 = self.conv15(x, edge_index15, w15)
        x16 = self.conv16(x, edge_index16, w16)
        x17 = self.conv17(x, edge_index17, w17)
        x18 = self.conv18(x, edge_index18, w18)
        x19 = self.conv19(x, edge_index19, w19)
        x20 = self.conv20(x, edge_index20, w20)
        x21 = self.conv21(x, edge_index21, w21)
        x22 = self.conv22(x, edge_index22, w22)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        x = F.log_softmax(x, dim=1)
        return x

class MGNN40(MGNN21):
    def __init__(self, in_channels, num_classes, hidden_channels=128):
        super(MGNN40, self).__init__(in_channels, num_classes, hidden_channels)
        self.conv21 = WOneLayerGCN(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(21, 40+1):
            self.convs.append(WOneLayerGCN(in_channels, hidden_channels))
