## Compositional Scene Representation Toolbox

This is an accompanied toolbox for the survey article: [Compositional Scene Representation Learning via Reconstruction: A Survey](https://arxiv.org/abs/2202.07135) \[Yuan et al., IEEE TPAMI 2023\]. The toolbox contains code for synthesizing multiple datasets that could be used for benchmarking compositional scene representation learning methods, and collects the implementations of the following papers:
- [Unsupervised Object-Centric Learning from Multiple Unspecified Viewpoints](https://arxiv.org/abs/2401.01922) \[Yuan et al., IEEE TPAMI-24\]
- [Time-Conditioned Generative Modeling of Object-Centric Representations for Video Decomposition and Prediction](https://arxiv.org/abs/2301.08951). \[Gao \& Li, UAI-23\]
- [Unsupervised Learning of Compositional Scene Representations from Multiple Unspecified Viewpoints](https://arxiv.org/abs/2112.03568) \[Yuan et al., AAAI-22\]
- [Knowledge-Guided Object Discovery with Acquired Deep Impressions](https://arxiv.org/abs/2103.10611) \[Yuan et al., AAAI-21\]
- [Generative Modeling of Infinite Occluded Objects for Compositional Scene Representation](https://proceedings.mlr.press/v97/yuan19b.html) \[Yuan et al., ICML-19\]
- [Spatial Mixture Models with Learnable Deep Priors for Perceptual Grouping](https://arxiv.org/abs/1902.02502) \[Yuan et al., AAAI-19\]

- [GENESIS-V2: Inferring Unordered Object Representations without Iterative Refinement](https://proceedings.neurips.cc/paper/2021/hash/43ec517d68b6edd3015b3edc9a11367b-Abstract.html) \[Engelcke et al., NeurIPS-21\]
- [Efficient Iterative Amortized Inference for Learning Symmetric and Disentangled Multi-Object Representations](https://proceedings.mlr.press/v139/emami21a.html) \[Emami et al., ICML-21\]
- [Object-Centric Learning with Slot Attention](https://papers.nips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) \[Locatello et al., NeurIPS-20\]
- [Learning Object-Centric Representations of Multi-Object Scenes from Multiple Views](https://papers.nips.cc/paper/2020/hash/3d9dabe52805a1ea21864b09f3397593-Abstract.html) \[Li et al., NeurIPS-20\]
- [GENESIS: Generative Scene Inference and Sampling with Object-Centric Latent Representations](https://openreview.net/forum?id=BkxfaTVFwH) \[Engelcke et al., ICLR-20\]
- [SPACE: Unsupervised Object-Oriented Scene Representation via Spatial Attention and Decomposition](https://openreview.net/forum?id=rkl03ySYDH) \[Lin et al., ICLR-20\]
- [MONet: Unsupervised Scene Decomposition and Representation](https://arxiv.org/abs/1901.11390) \[Burgess et al.\]
- [Multi-Object Representation Learning with Iterative Variational Inference](https://proceedings.mlr.press/v97/greff19a.html) \[Greff et al., ICML-19\]
- [Neural Expectation Maximization](https://papers.nips.cc/paper/2017/hash/d2cd33e9c0236a8c2d8bd3fa91ad3acf-Abstract.html) \[Greff et al., NeurIPS-17\]
- [Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](https://papers.nips.cc/paper/2016/hash/52947e0ade57a09e4a1386d08f17b656-Abstract.html) \[Eslami et al., NeurIPS-16\]

The `README.md` file in each folder contains the instructions on how to run the code.

### Submodules

Initialize submodules using the following command.
```bash
git submodule update --init --recursive
```

### Create Benchmark Datasets

Change the current working directory to `compositional-scene-representation-datasets` and follow the instructions described in `README.md` to create benchmark datasets.

### Evaluate Performance on Benchmark Datasets

#### AIR

Change the current working directory to `air-unofficial/experiments_benchmark` and run `run.sh` and `run_nc.sh`.

```bash
cd air-unofficial/experiments_benchmark
./run.sh
./run_nc.sh
cd ../..
```

Run `air-unofficial/experiments_benchmark/evaluate.ipynb` to evaluate the trained models.

#### N-EM

Change the current working directory to `nem-unofficial/experiments_benchmark` and run `run.sh` and `run_nc.sh`.

```bash
cd nem-unofficial/experiments_benchmark
./run.sh
./run_nc.sh
cd ../..
```

Run `nem-unofficial/experiments_benchmark/evaluate.ipynb` to evaluate the trained models.

#### IODINE

Change the current working directory to `iodine-unofficial/experiments_benchmark` and run `run.sh` and `run_nc.sh`.

```bash
cd iodine-unofficial/experiments_benchmark
./run.sh
./run_nc.sh
cd ../..
```

Run `iodine-unofficial/experiments_benchmark/evaluate.ipynb` to evaluate the trained models.

#### GMIOO

Change the current working directory to `infinite-occluded-objects/experiments_benchmark` and run `run.sh` and `run_nc.sh`.

```bash
cd infinite-occluded-objects/experiments_benchmark
./run.sh
./run_nc.sh
cd ../..
```

Run `infinite-occluded-objects/experiments_benchmark/evaluate.ipynb` to evaluate the trained models.

#### MONet

Change the current working directory to `monet-unofficial/experiments_benchmark` and run `run.sh`.

```bash
cd monet-unofficial/experiments_benchmark
./run.sh
cd ../..
```

Change the current working directory to `monet-unofficial_nc/experiments_benchmark` and run `run.sh`.

```bash
cd monet-unofficial_nc/experiments_benchmark
./run.sh
cd ../..
```

Run `monet-unofficial/experiments_benchmark/evaluate.ipynb` to evaluate the trained models.

#### SPACE

Change the current working directory to `SPACE/src` and run `run.sh`.

```bash
cd SPACE/src
./run.sh
cd ../..
```

Change the current working directory to `SPACE_nc/src` and run `run.sh`.

```bash
cd SPACE_nc/src
./run.sh
cd ../..
```

Run `SPACE/evaluate.ipynb` to evaluate the trained models.

#### Slot Attention

Change the current working directory to `slot-attention-unofficial/experiments_benchmark` and run `run.sh` and `run_nc.sh`.

```bash
cd slot-attention-unofficial/experiments_benchmark
./run.sh
./run_nc.sh
cd ../..
```

Run `slot-attention-unofficial/experiments_benchmark/evaluate.ipynb` to evaluate the trained models.

#### EfficientMORL

Change the current working directory to `EfficientMORL` and run `run.sh` and `run_nc.sh`.

```bash
cd EfficientMORL
./run.sh
./run_nc.sh
cd ..
```

Run `EfficientMORL/evaluate.ipynb` to evaluate the trained models.

#### GENESIS and GENESIS-V2

Change the current working directory to `genesis` and run `run.sh`.

```bash
cd genesis
./run.sh
cd ..
```

Change the current working directory to `genesis_nc` and run `run.sh`.

```bash
cd genesis_nc
./run.sh
cd ..
```

Run `genesis/evaluate.ipynb` to evaluate the trained models.
