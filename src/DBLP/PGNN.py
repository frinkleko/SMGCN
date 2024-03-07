from torch_geometric.seed import seed_everything

seed_everything(42)
from utils import *
import pickle

data = pickle.load(open('data/DBLP.pkl', 'rb'))
num_classes = data.num_classes

model = PGNN(data['author'].x.size(-1), num_classes)
model.to(device)

from torch_geometric.profile import count_parameters
from sklearn.metrics import f1_score, normalized_mutual_info_score

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

edge_index1 = data.view1_edge_index
edge_index2 = data.view2_edge_index
edge_index3 = data.view3_edge_index

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
        out = model(data['author'].x, edge_index1, edge_index2, edge_index3)
        loss = F.nll_loss(out[train_mask], data['author'].y[train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        val_loss = F.nll_loss(out[val_mask], data['author'].y[val_mask])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            wait = 0
        else:
            wait += 1
            if wait == 50:
                print('Early stopping!')
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
    _, pred = model(data['author'].x, edge_index1, edge_index2,
                    edge_index3).max(dim=1)
    np.save('labels/PGNN.npy', pred.cpu().numpy())
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
