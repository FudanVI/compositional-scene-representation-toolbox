## Multi-Object Network (MONet)

This is an <em>unofficial</em> implementation of the paper ["MONet: Unsupervised Scene Decomposition and Representation"](https://arxiv.org/abs/1901.11390).

### Dependencies

- tensorflow == 1.15
- dm-sonnet == 1.36
- numpy == 1.18
- h5py == 2.10
- pyyaml == 5.3
- scipy == 1.4
- scikit-learn == 0.22

### Datasets

Change the current working directory to `data` and run `create_datasets.sh`.

```bash
cd data
./create_datasets.sh
cd ..
```

### Experiments

Change the current working directory to `experiments` and run `run.sh`.

```bash
cd experiments
./run.sh
cd ..
```

Run `experiments/evaluate.ipynb` to evaluate the trained models.
