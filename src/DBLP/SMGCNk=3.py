from torch_geometric.seed import seed_everything

seed_everything(42)
from utils import *
import pickle

print('DBLP dataset')
data = pickle.load(open('data/DBLP.pkl', 'rb'))
num_classes = data.num_classes

model = MGNN22(data['author'].x.size(-1), num_classes)
from torch_geometric.profile import count_parameters
print('Number of parameters is {}'.format(count_parameters(model)))

model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)

num_nodes = data['author'].x.size(0)

# S
edge_index1 = data.view1_edge_index
w1 = torch.ones(edge_index1.size(1)).to(device)
edge_index2 = data.view2_edge_index
w2 = torch.ones(edge_index2.size(1)).to(device)
edge_index3 = data.view3_edge_index
w3 = torch.ones(edge_index3.size(1)).to(device)
agree_edge = data.agree_edge_index
w_agree = torch.ones(agree_edge.size(1)).to(device)

T_index1 = T2(edge_index1, num_nodes, self_loop=True)
T_index2 = T2(edge_index2, num_nodes, self_loop=True)
T_index3 = T2(edge_index3, num_nodes, self_loop=True)
print('A[i,j]=1 if i is nn of j or j is nn of i')
print('View 1 NN edge number', T_index1.nonzero()[0].shape)
print('View 2 NN edge number', T_index2.nonzero()[0].shape)
print('View 3 NN edge number', T_index3.nonzero()[0].shape)

T2_index = agreeT2([T_index1, T_index2, T_index3])
T2_index, wt2 = adj2index(T2_index)
agree_2, w_agree2 = power_sparse(agree_edge, num_nodes)

print('T1 edge index', agree_edge.size())
print('T2 edge index', T2_index.size())

print('edge index 1', edge_index1.size())
print('edge index 2', edge_index2.size())
print('edge index 3', edge_index3.size())

# S*T2
edge_index4, w4 = mm_sparse_w(edge_index1, T2_index, w1, wt2, num_nodes)
edge_index5, w5 = mm_sparse_w(edge_index2, T2_index, w2, wt2, num_nodes)
edge_index6, w6 = mm_sparse_w(edge_index3, T2_index, w3, wt2, num_nodes)

index1_, w1_ = power_sparse(edge_index1, num_nodes)
index2_, w2_ = power_sparse(edge_index2, num_nodes)
index3_, w3_ = power_sparse(edge_index3, num_nodes)

# S*S*T2
edge_index7, w7 = mm_sparse_w(index1_, T2_index, w1_, wt2, num_nodes)
edge_index8, w8 = mm_sparse_w(index2_, T2_index, w2_, wt2, num_nodes)
edge_index9, w9 = mm_sparse_w(index3_, T2_index, w3_, wt2, num_nodes)

# S*T1
edge_index10, w10 = mm_sparse_w(edge_index1, agree_edge, w1, w_agree,
                                    num_nodes)
edge_index11, w11 = mm_sparse_w(edge_index2, agree_edge, w2, w_agree,
                                    num_nodes)
edge_index12, w12 = mm_sparse_w(edge_index3, agree_edge, w3, w_agree,
                                    num_nodes)

# S*T1*T1
edge_index13, w13 = mm_sparse_w(edge_index1, agree_2, w1, w_agree2,
                                    num_nodes)
edge_index14, w14 = mm_sparse_w(edge_index2, agree_2, w2, w_agree2,
                                    num_nodes)
edge_index15, w15 = mm_sparse_w(edge_index3, agree_2, w3, w_agree2,
                                    num_nodes)

T22, wt22 = power_sparse(T2_index, num_nodes)

# S*T2*T2
edge_index16, w16 = mm_sparse_w(edge_index1, T22, w1, wt22, num_nodes)
edge_index17, w17 = mm_sparse_w(edge_index2, T22, w2, wt22, num_nodes)
edge_index18, w18 = mm_sparse_w(edge_index3, T22, w3, wt22, num_nodes)

# S*S*T1
edge_index19, w19 = mm_sparse_w(index1_, agree_edge, w1_, w_agree,
                                    num_nodes)
edge_index20, w20 = mm_sparse_w(index2_, agree_edge, w2_, w_agree,
                                    num_nodes)
edge_index21, w21 = mm_sparse_w(index3_, agree_edge, w3_, w_agree,
                                    num_nodes)

edge_index_id = torch.tensor(np.arange(data['author'].x.size(0)),
                             dtype=torch.long).unsqueeze(0).repeat(
                                 2, 1).to(device)
w_id = torch.ones(edge_index_id.size(1)).to(device)

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

records = []

# 6
model = MGNN22(data['author'].x.size(-1), num_classes)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
for run in range(1):
    import copy
    # seed_everything(42)
    best_val_loss = 100
    best_model = None
    wait = 0
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        out = model(data['author'].x, edge_index_id, edge_index1, edge_index2,
                    edge_index3, edge_index4, edge_index5, edge_index6,
                    edge_index7, edge_index8, edge_index9, edge_index10,
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21)
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
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21).max(dim=1)
    correct = float(pred[data['author'].test_mask].eq(
        data['author'].y[data['author'].test_mask]).sum().item())
    acc = correct / data['author'].test_mask.sum().item()
    print('Test Accuracy: {:.5f}'.format(acc))
    records.append(acc)
print('Average accuracy is {}'.format(np.round(np.mean(records) * 100, 2)))
print('Std is {}'.format(np.round(np.std(records) * 100, 2)))

# 7
model = MGNN22(data['author'].x.size(-1), num_classes)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
for run in range(1):
    import copy
    # seed_everything(42)
    best_val_loss = 100
    best_model = None
    wait = 0
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        out = model(data['author'].x, edge_index_id, edge_index1, edge_index2,
                    edge_index3, edge_index4, edge_index5, edge_index6,
                    edge_index7, edge_index8, edge_index9, edge_index10,
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21)
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
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21).max(dim=1)
    correct = float(pred[data['author'].test_mask].eq(
        data['author'].y[data['author'].test_mask]).sum().item())
    acc = correct / data['author'].test_mask.sum().item()
    print('Test Accuracy: {:.5f}'.format(acc))
    records.append(acc)
print('Average accuracy is {}'.format(np.round(np.mean(records) * 100, 2)))
print('Std is {}'.format(np.round(np.std(records) * 100, 2)))

# 8
model = MGNN22(data['author'].x.size(-1), num_classes)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
for run in range(1):
    import copy
    # seed_everything(42)
    best_val_loss = 100
    best_model = None
    wait = 0
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        out = model(data['author'].x, edge_index_id, edge_index1, edge_index2,
                    edge_index3, edge_index4, edge_index5, edge_index6,
                    edge_index7, edge_index8, edge_index9, edge_index10,
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21)
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
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21).max(dim=1)
    correct = float(pred[data['author'].test_mask].eq(
        data['author'].y[data['author'].test_mask]).sum().item())
    acc = correct / data['author'].test_mask.sum().item()
    print('Test Accuracy: {:.5f}'.format(acc))
    records.append(acc)
print('Average accuracy is {}'.format(np.round(np.mean(records) * 100, 2)))
print('Std is {}'.format(np.round(np.std(records) * 100, 2)))

# 9
model = MGNN22(data['author'].x.size(-1), num_classes)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
for run in range(1):
    import copy
    # seed_everything(42)
    best_val_loss = 100
    best_model = None
    wait = 0
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        out = model(data['author'].x, edge_index_id, edge_index1, edge_index2,
                    edge_index3, edge_index4, edge_index5, edge_index6,
                    edge_index7, edge_index8, edge_index9, edge_index10,
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21)
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
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21).max(dim=1)
    correct = float(pred[data['author'].test_mask].eq(
        data['author'].y[data['author'].test_mask]).sum().item())
    acc = correct / data['author'].test_mask.sum().item()
    print('Test Accuracy: {:.5f}'.format(acc))
    records.append(acc)
print('Average accuracy is {}'.format(np.round(np.mean(records) * 100, 2)))
print('Std is {}'.format(np.round(np.std(records) * 100, 2)))

# 10
model = MGNN22(data['author'].x.size(-1), num_classes)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
for run in range(1):
    import copy
    # seed_everything(42)
    best_val_loss = 100
    best_model = None
    wait = 0
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        out = model(data['author'].x, edge_index_id, edge_index1, edge_index2,
                    edge_index3, edge_index4, edge_index5, edge_index6,
                    edge_index7, edge_index8, edge_index9, edge_index10,
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21)
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
                    edge_index11, edge_index12, edge_index13, edge_index14,
                    edge_index15, edge_index16, edge_index17, edge_index18,
                    edge_index19, edge_index20, edge_index21, w_id, w1, w2, w3,
                    w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16,
                    w17, w18, w19, w20, w21).max(dim=1)
    correct = float(pred[data['author'].test_mask].eq(
        data['author'].y[data['author'].test_mask]).sum().item())
    acc = correct / data['author'].test_mask.sum().item()
    print('Test Accuracy: {:.5f}'.format(acc))
    records.append(acc)
print('Average accuracy is {}'.format(np.round(np.mean(records) * 100, 2)))
print('Std is {}'.format(np.round(np.std(records) * 100, 2)))
