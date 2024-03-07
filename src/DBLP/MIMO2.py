from torch_geometric.seed import seed_everything

seed_everything(42)
from utils import *
import pickle
from sklearn.metrics import f1_score, normalized_mutual_info_score

data = pickle.load(open('data/DBLP.pkl', 'rb'))
num_classes = data.num_classes

model = MGNN13(data['author'].x.size(1), num_classes)
model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

num_nodes = data['author'].x.size(0)
edge_index1 = data.view1_edge_index
w1 = torch.ones(edge_index1.size(1)).to(device)
edge_index2 = data.view2_edge_index
w2 = torch.ones(edge_index2.size(1)).to(device)
edge_index3 = data.view3_edge_index
w3 = torch.ones(edge_index3.size(1)).to(device)

edge_index_id = torch.tensor(np.arange(data['author'].x.size(0)),
                             dtype=torch.long).unsqueeze(0).repeat(
                                 2, 1).to(device)

edge_index4, w4 = mm_sparse_w(edge_index1, edge_index2, w1, w2, num_nodes)
edge_index5, w5 = mm_sparse_w(edge_index1, edge_index3, w1, w3, num_nodes)
edge_index6, w6 = mm_sparse_w(edge_index1, edge_index1, w1, w1, num_nodes)
edge_index7, w7 = mm_sparse_w(edge_index2, edge_index3, w2, w3, num_nodes)
edge_index8, w8 = mm_sparse_w(edge_index2, edge_index1, w2, w1, num_nodes)
edge_index9, w9 = mm_sparse_w(edge_index2, edge_index2, w2, w2, num_nodes)
edge_index10, w10 = mm_sparse_w(edge_index3, edge_index1, w3, w1, num_nodes)
edge_index11, w11 = mm_sparse_w(edge_index3, edge_index2, w3, w2, num_nodes)
edge_index12, w12 = mm_sparse_w(edge_index3, edge_index3, w3, w3, num_nodes)

w_id = torch.ones(edge_index_id.size(1)).to(device)

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

records = []
for run in range(1):
    import copy
    seed_everything(42)
    best_val_loss = 100
    best_model = None
    wait = 0
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        out = model(data['author'].x, edge_index_id, edge_index1, edge_index2,
                    edge_index3, edge_index4, edge_index5, edge_index6,
                    edge_index7, edge_index8, edge_index9, edge_index10,
                    edge_index11, edge_index12, w_id, w1, w2, w3, w4, w5, w6,
                    w7, w8, w9, w10, w11, w12)
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
            # print('Epoch: {:03d}, Loss: {:.5f}, Test Acc: {:.5f}'.format(
            #     epoch, loss, acc))
    model = best_model
    model.eval()
    _, pred = model(data['author'].x, edge_index_id, edge_index1, edge_index2,
                    edge_index3, edge_index4, edge_index5, edge_index6,
                    edge_index7, edge_index8, edge_index9, edge_index10,
                    edge_index11, edge_index12, w_id, w1, w2, w3, w4, w5, w6,
                    w7, w8, w9, w10, w11, w12).max(dim=1)
    np.save('labels/MIMO2.npy', pred.cpu().numpy())
    correct = float(pred[data['author'].test_mask].eq(
        data['author'].y[data['author'].test_mask]).sum().item())
    acc = correct / data['author'].test_mask.sum().item()
    f1 = f1_score(data['author'].y[data['author'].test_mask].cpu().numpy(),
                  pred[data['author'].test_mask].cpu().numpy(),
                  average='macro')
    nmi = normalized_mutual_info_score(
        data['author'].y[data['author'].test_mask].cpu().numpy(),
        pred[data['author'].test_mask].cpu().numpy())
    print('Test Accuracy: {:.5f}'.format(acc))
    print('Test F1: {:.5f}'.format(f1))
    print('Test NMI: {:.5f}'.format(nmi))
    records.append(acc)
print('Average accuracy is {}'.format(np.round(np.mean(records) * 100, 2)))
print('Std is {}'.format(np.round(np.std(records) * 100, 2)))
