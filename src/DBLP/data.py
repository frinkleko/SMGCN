from torch_geometric.seed import seed_everything
seed_everything(42)
from utils import *
import torch
from itertools import combinations


name = 'DBLP'
data =  load_HGBD_data(name)

edge_index_author_to_paper = data.edge_index_dict[('author', 'to', 'paper')]
edge_index_paper_to_conference = data.edge_index_dict[('paper', 'to', 'conference')]
edge_index_paper_to_term = data.edge_index_dict[('paper', 'to', 'term')]

# View 1
paper_to_authors = {}
for author, paper in edge_index_author_to_paper.t().tolist():
    if paper not in paper_to_authors:
        paper_to_authors[paper] = []
    paper_to_authors[paper].append(author)

view1_edges = [list(combinations(authors, 2)) for authors in paper_to_authors.values() if len(authors) > 1]
view1_edges = torch.tensor([edge for edges in view1_edges for edge in edges], dtype=torch.long).t().contiguous()

# View 2
conference_to_authors = {}
for paper, conference in edge_index_paper_to_conference.t().tolist():
    if conference not in conference_to_authors:
        conference_to_authors[conference] = []
    conference_to_authors[conference].extend(paper_to_authors.get(paper, []))

view2_edges = [list(combinations(set(authors), 2)) for authors in conference_to_authors.values() if len(authors) > 1]
view2_edges = torch.tensor([edge for edges in view2_edges for edge in edges], dtype=torch.long).t().contiguous()

# View 3
term_to_authors = {}
for paper, term in edge_index_paper_to_term.t().tolist():
    if term not in term_to_authors:
        term_to_authors[term] = []
    term_to_authors[term].extend(paper_to_authors.get(paper, []))

view3_edges = [list(combinations(set(authors), 2)) for authors in term_to_authors.values() if len(authors) > 1]
view3_edges = torch.tensor([edge for edges in view3_edges for edge in edges], dtype=torch.long).t().contiguous()

agree_edge_index = agree_edge([view1_edges, view2_edges, view3_edges])
print('agree edge index has {} edges'.format(agree_edge_index.shape[1]))

data.view1_edge_index = view1_edges
print('view1 edge index has {} edges'.format(view1_edges.shape[1]))
data.view2_edge_index = view2_edges
print('view2 edge index has {} edges'.format(view2_edges.shape[1]))
data.view3_edge_index = view3_edges
print('view3 edge index has {} edges'.format(view3_edges.shape[1]))
data.agree_edge_index = agree_edge_index

num_classes = int(data['author'].y.max() + 1)
data.num_classes = num_classes

train_mask = data['author'].train_mask
idx = np.where(train_mask.cpu().numpy() == 1)[0]
np.random.shuffle(idx)
val_mask = torch.zeros(train_mask.shape[0], dtype=torch.bool)
val_mask[idx[:int(len(idx) * 0.1)]] = 1
train_mask[idx[:int(len(idx) * 0.1)]] = 0

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = data['author'].test_mask

test_mask = data['author'].test_mask
print('train: {}, val: {},  test: {}'.format(train_mask.sum(), val_mask.sum(), test_mask.sum()))

import pickle
with open('./data/DBLP.pkl', 'wb') as f:
    pickle.dump(data, f)