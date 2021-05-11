## Neural Expectation Maximization (N-EM)

This is an <em>unofficial</em> implementation of the paper ["Neural Expectation Maximization"](https://papers.nips.cc/paper/7246-neural-expectation-maximization). The <em>official</em> implementation is <https://github.com/sjoerdvansteenkiste/Neural-EM>.

### Dependencies

- pytorch == 1.8
- numpy == 1.19
- h5py == 2.10
- pyyaml == 5.4
- scipy == 1.6
- scikit-learn == 0.24

### Datasets

Change the current working directory to `data` and run `create_datasets.sh`. Some of the datasets provided by the [official repository](https://github.com/sjoerdvansteenkiste/Neural-EM) will be downloaded and converted.

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
