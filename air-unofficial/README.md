## Attend-Infer-Repeat (AIR)

This is an <em>unofficial</em> implementation of the paper ["Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"](https://papers.nips.cc/paper/6230-attend-infer-repeat-fast-scene-understanding-with-generative-models).

### Dependencies

- pytorch == 1.8
- torchvision == 0.9
- numpy == 1.19
- h5py == 2.10
- pyyaml == 5.4
- scipy == 1.6
- scikit-learn == 0.24
- scikit-image == 0.17

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
