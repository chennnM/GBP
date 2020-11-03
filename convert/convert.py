import pickle as pkl
import sys
import os
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing
import json
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
import time
import scipy.sparse
from sklearn.preprocessing import StandardScaler

def graphsave(adj,dir):
    adj = adj.tocoo()
    f=open(dir,'w')
    n = str(adj.shape[0])+'\n'
    f.write(n)
    for u,v in zip(adj.row,adj.col):
        m=str(u)+' '+str(v)+'\n'
        f.write(m)
    f.close()


def load_graphsage_data(dataset_path, dataset_str, normalize=True):
  """Load GraphSAGE data."""
  start_time = time.time()

  graph_json = json.load(
      gfile.Open('{}/{}/{}-G.json'.format(dataset_path, dataset_str,
                                          dataset_str)))
  graph_nx = json_graph.node_link_graph(graph_json)

  id_map = json.load(
      gfile.Open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str,
                                               dataset_str)))
  is_digit = list(id_map.keys())[0].isdigit()
  id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
  class_map = json.load(
      gfile.Open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str,
                                                  dataset_str)))

  is_instance = isinstance(list(class_map.values())[0], list)
  class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
               for k, v in class_map.items()}

  broken_count = 0
  to_remove = []
  for node in graph_nx.nodes():
    if node not in id_map:
      to_remove.append(node)
      broken_count += 1
  for node in to_remove:
    graph_nx.remove_node(node)
  tf.logging.info(
      'Removed %d nodes that lacked proper annotations due to networkx versioning issues',
      broken_count)

  feats = np.load(
      gfile.Open(
          '{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str),
          'rb'))

  tf.logging.info('Loaded data (%f seconds).. now preprocessing..',
                  time.time() - start_time)
  start_time = time.time()

  edges = []
  for edge in graph_nx.edges():
    if edge[0] in id_map and edge[1] in id_map:
      edges.append((id_map[edge[0]], id_map[edge[1]]))
  num_data = len(id_map)

  val_data = np.array(
      [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]['val']],
      dtype=np.int32)
  test_data = np.array(
      [id_map[n] for n in graph_nx.nodes() if graph_nx.node[n]['test']],
      dtype=np.int32)
  is_train = np.ones((num_data), dtype=np.bool)
  is_train[val_data] = False
  is_train[test_data] = False
  train_data = np.array([n for n in range(num_data) if is_train[n]],
                        dtype=np.int32)

  train_edges = [
      (e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]
  ]
  edges = np.array(edges, dtype=np.int32)
  train_edges = np.array(train_edges, dtype=np.int32)

  # Process labels
  if isinstance(list(class_map.values())[0], list):
    num_classes = len(list(class_map.values())[0])
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for k in class_map.keys():
      labels[id_map[k], :] = np.array(class_map[k])
  else:
    num_classes = len(set(class_map.values()))
    labels = np.zeros((num_data, num_classes), dtype=np.float32)
    for k in class_map.keys():
      labels[id_map[k], class_map[k]] = 1

  if normalize:
    train_ids = np.array([
        id_map[n]
        for n in graph_nx.nodes()
        if not graph_nx.node[n]['val'] and not graph_nx.node[n]['test']
    ])
    train_feats = feats[train_ids]
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

  def _construct_adj(edges):
    adj = sp.csr_matrix((np.ones(
        (edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
                        shape=(num_data, num_data))
    # adj += adj.transpose()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj

  train_adj = _construct_adj(train_edges)
  full_adj = _construct_adj(edges)

  train_feats = feats[train_data]
  test_feats = feats
  if dataset_str=='Amazon2M':
    test_data = val_data

  tf.logging.info('Data loaded, %f seconds.', time.time() - start_time)
  return num_data, train_adj, full_adj, feats, train_feats, test_feats, labels, train_data, val_data, test_data

def load_data(prefix, normalize=True):
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1

    node_train = np.array(role['tr'])
    node_val = np.array(role['va'])
    node_test = np.array(role['te'])
    train_feats = feats[node_train]
    adj_train = adj_train[node_train,:][:,node_train]
    labels = class_arr
    return adj_full, adj_train, feats, train_feats, labels, node_train, node_val, node_test


def Yelp(datastr='yelp'):
    adj_full, adj_train, feats, train_feats, labels, idx_train, idx_val, idx_test = load_data(datastr)
    graphsave(adj_full,dir=datastr+'.txt')
    graphsave(adj_train,dir=datastr+'_train.txt')
    np.save(datastr+'_feat.npy',feats)
    np.save(datastr+'_train_feat.npy',train_feats)
    np.savez(datastr+'_labels.npz',labels=labels,idx_train=idx_train,idx_val=idx_val,idx_test=idx_test)

def Amazon2M(dataset='Amazon2M'):
    _, train_adj, full_adj, feats, train_feats, _, labels, idx_train, idx_val, idx_test = load_graphsage_data('.', dataset, normalize=True)
    train_adj = train_adj[idx_train, :][:, idx_train]
    labels = np.where(labels>0.5)[1]
    graphsave(full_adj,dir=dataset+'.txt')
    graphsave(train_adj,dir=dataset+'_train.txt')
    np.save(dataset+'_feat.npy',feats)A
    np.save(dataset+'_train_feat.npy',train_feats)
    np.savez(dataset+'_labels.npz',labels=labels,idx_train=idx_train,idx_val=idx_val,idx_test=idx_test)

def PPI(dataset='ppi'):
    _, train_adj, full_adj, feats, train_feats, _, labels, idx_train, idx_val, idx_test = load_graphsage_data('.', 'ppi', normalize=True)
    train_adj = train_adj[idx_train, :][:, idx_train]
    graphsave(full_adj,dir=dataset+'.txt')
    graphsave(train_adj,dir=dataset+'_train.txt')
    np.save(dataset+'_feat.npy',feats)
    np.save(dataset+'_train_feat.npy',train_feats)
    np.savez(dataset+'_labels.npz',labels=labels,idx_train=idx_train,idx_val=idx_val,idx_test=idx_test)

