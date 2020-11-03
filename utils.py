import numpy as np
import BiP
import torch
from sklearn.metrics import f1_score
import gc

def load_inductive(datastr,alpha,rmax,rrz):
    features_train = BiP.ppr(datastr+'_train',alpha,rmax,rrz)
    features = BiP.ppr(datastr,alpha,rmax,rrz)
    features_train = torch.FloatTensor(features_train).T
    features = torch.FloatTensor(features).T
    data = np.load("data/"+datastr+"_labels.npz")
    labels = data['labels']
    idx_train = data['idx_train']
    idx_val = data['idx_val']
    idx_test = data['idx_test']
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return features_train,features,labels,idx_train,idx_val,idx_test

def load_citation(datastr,alpha,rmax,rrz):
    features = BiP.ppr(datastr,alpha,rmax,rrz)
    features = torch.FloatTensor(features).T
    data = np.load("data/"+datastr+"_labels.npz")
    labels = data['labels']
    idx_train = data['idx_train']
    idx_val = data['idx_val']
    idx_test = data['idx_test']
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    features_train = features[idx_train]
    return features,labels,idx_train,idx_val,idx_test

def load_friendster(datastr='friendster',rmax=4e-8,rwnum=10000,rrz=0.5):
    features_train,features_val,features_test = pre_friendster(datastr,rmax,rwnum,rrz)
    features_train = features_train.T
    features_val = features_val.T
    features_test = features_test.T
    data = np.load("data/"+datastr+"_labels.npz")
    train_labels = torch.LongTensor(data['train_labels'])
    val_labels = torch.LongTensor(data['val_labels'])
    test_labels = torch.LongTensor(data['test_labels'])
    return features_train,features_val,features_test,train_labels,val_labels,test_labels

def pre_friendster(datastr='friendster',rmax=4e-8,rwnum=10000,rrz=0.5):
    features = BiP.transition(datastr,rmax,rwnum,rrz)
    train_idx = np.load("data/"+datastr+"_labels.npz")['train_idx']
    val_idx = np.load("data/"+datastr+"_labels.npz")['val_idx']
    test_idx = np.load("data/"+datastr+"_labels.npz")['test_idx']
    tmp = np.array(features[0],dtype=np.float32)
    feat_train = tmp[train_idx]
    feat_val = tmp[val_idx]
    feat_test = tmp[test_idx]
    for i in range(1,100):
        tmp = np.array(features[i],dtype=np.float32)
        feat_train = np.vstack((feat_train,tmp[train_idx]))
        feat_val = np.vstack((feat_val,tmp[val_idx]))
        feat_test = np.vstack((feat_test,tmp[test_idx]))

    del features
    gc.collect()
    return feat_train,feat_val,feat_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def muticlass_f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro

def mutilabel_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return f1_score(y_true, y_pred, average="micro")