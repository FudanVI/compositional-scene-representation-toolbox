#!/bin/bash

export GIT_LFS_SKIP_SMUDGE=1
git submodule update --init --recursive
cd ocloc-data
for name in clevr shop gso shapenet; do
    git lfs pull -I ${name}.h5
done
cd ..
unset GIT_LFS_SKIP_SMUDGE
