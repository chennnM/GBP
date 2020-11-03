from __future__ import division
from __future__ import print_function
from utils import load_friendster,muticlass_f1
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import GnnBP
import uuid

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=20159, help='random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
parser.add_argument('--layer', type=int, default=4, help='number of layers.')
parser.add_argument('--hidden', type=int, default=128, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate.')
parser.add_argument('--patience', type=int, default=50, help='patience')
parser.add_argument('--data', default='friendster', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--rwnum', type=int, default=10000, help='random walk')
parser.add_argument('--rmax', type=float, default=4e-8, help='threshold.')
parser.add_argument('--rrz', type=float, default=0.5, help='r.')
parser.add_argument('--bias', default='bn', help='bias.')
parser.add_argument('--batch', type=int, default=2048, help='batch size')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("--------------------------")
print(args)
features_train,features_val,features_test,train_labels,val_labels,test_labels = load_friendster(datastr=args.data,rmax=args.rmax,rwnum=args.rwnum,rrz=args.rrz)

features_train = torch.FloatTensor(features_train).cuda(args.dev)
features_val = torch.FloatTensor(features_val).cuda(args.dev)
features_test = torch.FloatTensor(features_test).cuda(args.dev)

label_dim = int(max(train_labels.max(),val_labels.max(),test_labels.max()))+1

train_labels = train_labels.cuda(args.dev)
val_labels = val_labels.cuda(args.dev)
test_labels = test_labels.cuda(args.dev)


checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

model = GnnBP(nfeat=features_train.shape[1],
            nlayers=args.layer,
            nhidden=args.hidden,
            nclass=label_dim,
            dropout=args.dropout,
            bias = args.bias).cuda(args.dev)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.CrossEntropyLoss()

torch_dataset = Data.TensorDataset(features_train, train_labels)
loader = Data.DataLoader(dataset=torch_dataset,
                        batch_size=args.batch,
                        shuffle=True,
                        num_workers=0)

def train():
    model.train()
    loss_list = []
    time_epoch = 0
    for step, (batch_x, batch_y) in enumerate(loader):
        # batch_x = batch_x.cuda(args.dev)
        # batch_y = batch_y.cuda(args.dev)
        t1 = time.time()
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        time_epoch+=(time.time()-t1)
        loss_list.append(loss_train.item())
    return np.mean(loss_list),time_epoch

def validate():
    model.eval()
    with torch.no_grad():
        output = model(features_val)
        micro_val = muticlass_f1(output, val_labels)
        return micro_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features_test)
        micro_test = muticlass_f1(output, test_labels)
        return micro_test.item()
    
bad_counter = 0
best = 0
best_epoch = 0
train_time = 0

for epoch in range(args.epochs):

    loss_tra,train_ep = train()
    f1_val = validate()
    train_time+=train_ep
    if(epoch+1)%50 == 0: 
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                '| val',
                'acc:{:.3f}'.format(f1_val),
                '| cost{:.3f}'.format(train_time))
    if f1_val > best:
        best = f1_val
        best_epoch = epoch
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

f1_test = test()

print("Train cost: {:.4f}s".format(train_time))
print('Load {}th epoch'.format(best_epoch))
print("Test f1:{:.3f}".format(f1_test))
print("--------------------------")
    





