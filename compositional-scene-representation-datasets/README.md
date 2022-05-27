## Compositional Scene Representation Datasets

This repository contains code for synthesizing multiple datasets that could be used to benchmark compositional scene representation learning methods.

### Dependencies

- numpy == 1.20
- h5py == 3.2
- pyyaml == 5.4
- scikit-image == 0.18
- imageio == 2.9
- torchvision == 0.9
- tensorflow == 2.5

### Blender

Synthesize images in the CLEVR dataset using Blender. The tested version of Blender is 2.79.

```bash
cd clevr-dataset-gen/image_generation
./create_blend.sh
./rename_files.sh
./create_pngs.sh
cd ../..
```

Synthesize images in the SHOP-VRB dataset using Blender. The tested version of Blender is 2.83.

```bash
cd shop-vrb-gen/image_generation_single
./create_blend.sh
./rename_files.sh
./create_pngs.sh
cd ../..
```

### HDF5

Create datasets in the HDF5 format.

```bash
./create_datasets.sh
```

### TFRecord

Create datasets in the TFRecord format.

```bash
cd tf
./create_datasets.sh
cd ..
```
