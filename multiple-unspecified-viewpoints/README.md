## Object-Centric Learning with Object Constancy (OCLOC)

This is the code repository of the paper "Unsupervised Learning of Compositional Scene Representations from Multiple Unspecified Viewpoints".

### Dependencies

- pytorch == 1.8
- torchvision == 0.9
- numpy == 1.20
- h5py == 3.2
- pyyaml == 5.4
- tensorboard == 2.5
- scipy == 1.6
- scikit-learn == 0.24

### Datasets

Synthesize images in the CLEVR and CLEVR-M datasets using Blender. The tested version of Blender is 2.79.

```bash
cd clevr-dataset-gen/image_generation
./create_blend.sh
./rename_files.sh
./create_pngs.sh
cd ../image_generation_viewpoint_1
./create_blend.sh
./create_pngs.sh
cd ../image_generation_viewpoint_2
./create_blend.sh
./create_pngs.sh
cd ../..
```

Create datasets in the HDF5 format.

```bash
cd data
./create_datasets.sh
cd ..
```

### Experiments

Train and test models.

```bash
cd exp_multi
./run.sh
cd ../exp_single
./run.sh
cd ..
```

Run `exp_multi/compute_metrics.ipynb` and `exp_single/compute_metrics.ipynb` to evaluate the performance.
