# Scalable Graph Neural Networks via Bidirectional Propagation

This repository contains a PyTorch implementation of "Scalable Graph Neural Networks via Bidirectional Propagation".(https://arxiv.org/abs/2007.02133)

## Requirements
- CUDA 10.1.243
- python 3.6.10
- pytorch 1.4.0
- GCC 5.4.0
- [cnpy](https://github.com/rogersce/cnpy)
- [swig-4.0.1](https://github.com/swig/swig)

## Datasets

The `data` folder includes three benchmark datasets(Cora, Citeseer, Pubmed). 
Other datasets can be downloaded from [PPI](http://snap.stanford.edu/graphsage/), [Yelp](https://github.com/GraphSAINT/GraphSAINT), [Amazon2M](https://github.com/google-research/google-research/tree/master/cluster_gcn) and [Friendster](http://snap.stanford.edu/data/com-Friendster.html). We also provide code to convert datasets to our format (in `convert` folder).

## Compilation
```sh
make
```
## Running the code

- To replicate the transductive learning results (Cora, Citeseer, Pubmed), run the following script

```sh
sh transductive.sh
```

- To replicate the inductive learning results (PPI, Yelp, Amazon2M), run the following script

```sh
sh inductive.sh
```

- To replicate the inductive results of Friendster, run the following script
 
```sh
sh friendster.sh
```

## Citation
```
@article{cwdlydw2020gbp,
  title = {Scalable Graph Neural Networks via Bidirectional Propagation},
  author = {Ming Chen, Zhewei Wei, Bolin Ding, Yaliang Li, Ye Yuan, Xiaoyong Du and Ji-Rong Wen},
  year = {2020},
  booktitle = {{NeurIPS}},
}
```
