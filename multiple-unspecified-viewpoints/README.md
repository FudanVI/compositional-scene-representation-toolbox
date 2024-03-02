## Object-Centric Learning with Object Constancy (OCLOC)

This is the code repository of the papers ["Unsupervised Learning of Compositional Scene Representations from Multiple Unspecified Viewpoints"](https://doi.org/10.1609/aaai.v36i8.20880) (commit 49097be68007017935f70f172293b0df8c7c68e9) and ["Unsupervised Object-Centric Learning from Multiple Unspecified Viewpoints"](https://doi.org/10.1109/TPAMI.2023.3349174) (the latest commit).

### Dependencies

- pytorch == 2.1
- hydra-core == 1.3
- omegaconf == 2.3
- numpy == 1.26
- matplotlib == 3.8
- imageio == 2.31
- h5py == 3.9
- tensorboard == 2.12
- scipy == 1.11
- scikit-learn == 1.3

### Datasets

Download the datasets from https://huggingface.co/datasets/jinyangyuan/ocloc-data.

```bash
./download_datasets.sh
```

### Experiments

Train and test models.

```bash
cd exp_multi
./run.sh
cd ../exp_multi_no_shadow
./run.sh
cd ..
```

Run `exp_multi/evaluate.ipynb` and `exp_multi_no_shadow/evaluate.ipynb` to evaluate the performance.
