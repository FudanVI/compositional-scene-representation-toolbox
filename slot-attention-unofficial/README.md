## Slot Attention

This is an <em>unofficial</em> implementation of the paper ["Object-Centric Learning with Slot Attention"](https://papers.nips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html). The <em>official</em> implementation is <https://github.com/google-research/google-research/tree/master/slot_attention>.

### Dependencies

- tensorflow == 2.5
- numpy == 1.19
- h5py == 3.1
- pyyaml == 5.4
- scipy == 1.7
- scikit-learn == 0.24

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
