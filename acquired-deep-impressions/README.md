## Acquired Deep Impressions (ADI)

This is the code repository of the paper "Knowledge-Guided Object Discovery with Acquired Deep Impressions".

### Dependencies

- pytorch == 1.6
- torchvision == 0.7
- numpy == 1.19
- h5py == 2.8
- pyyaml == 5.3
- scipy == 1.5
- scikit-learn == 0.23

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
