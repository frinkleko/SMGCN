from torch_geometric.seed import seed_everything

seed_everything(42)
from utils import *
import pickle

data = pickle.load(open('data/DBLP.pkl', 'rb'))
num_classes = data.num_classes
num_nodes = data['author'].x.size(0)

model = MGNN7(data['author'].x.size(-1), num_classes)
model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

edge_index1 = data.view1_edge_index
w1 = torch.ones(edge_index1.size(1)).to(device)
edge_index2 = data.view2_edge_index
w2 = torch.ones(edge_index2.size(1)).to(device)
edge_index3 = data.view3_edge_index
w3 = torch.ones(edge_index3.size(1)).to(device)

edge_index4, w4 = mm_sparse_w(edge_index1, edge_index1, num_nodes)
edge_index5, w5 = mm_sparse_w(edge_index2, edge_index2, num_nodes)
edge_index6, w6 = mm_sparse_w(edge_index3, edge_index3, num_nodes)

edge_index_id = torch.tensor(np.arange(data['author'].x.size(0)),
                             dtype=torch.long).unsqueeze(0).repeat(
                                 2, 1).to(device)
w_id = torch.ones(edge_index_id.size(1)).to(device)
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

records = []
for run in range(1):
    import copy
    best_val_loss = 100
    best_model = None
    wait = 0
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        out = model(data['author'].x, edge_index1, edge_index2, edge_index3,
                    edge_index4, edge_index5, edge_index6, edge_index_id, w1,
                    w2, w3, w4, w5, w6, w_id)
        loss = F.nll_loss(out[train_mask], data['author'].y[train_mask])
        loss.backward()
        optimizer.step()
        val_loss = F.nll_loss(out[val_mask], data['author'].y[val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            wait = 0
        else:
            wait += 1
            if wait == 50:
                break
        if epoch % 10 == 0:
            _, pred = out.max(dim=1)
            correct = float(pred[data['author'].test_mask].eq(
                data['author'].y[data['author'].test_mask]).sum().item())
            acc = correct / data['author'].test_mask.sum().item()
            print('Epoch: {:03d}, Loss: {:.5f}, Test Acc: {:.5f}'.format(
                epoch, loss, acc))
    model = best_model
    model.eval()
    _, pred = model(data['author'].x, edge_index1, edge_index2, edge_index3,
                    edge_index4, edge_index5, edge_index6, edge_index_id, w1,
                    w2, w3, w4, w5, w6, w_id).max(dim=1)
    correct = float(pred[data['author'].test_mask].eq(
        data['author'].y[data['author'].test_mask]).sum().item())
    acc = correct / data['author'].test_mask.sum().item()
    print('Test Accuracy: {:.5f}'.format(acc))
    records.append(acc)
print('Average accuracy is {}'.format(np.round(np.mean(records) * 100, 2)))
print('Std is {}'.format(np.round(np.std(records) * 100, 2)))
