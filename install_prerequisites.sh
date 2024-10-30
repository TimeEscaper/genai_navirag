#!/usr/bin/env bash

# LLaVa
cd third_party/
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA/
pip install -e .

# Detectron2 and Detic
cd ..
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2/
pip3 install -e .
cd ..
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
pip3 install -r requirements.txt
cd third_party/CenterNet2/
cp ../../../../misc/setup_centernet.py setup.py
pip3 install -e .
cd ../../
cp ../../misc/setup_detic.py setup.py
DETIC_DIR="$(pwd)"
cd ../../

# SAM
pip3 install git+https://github.com/facebookresearch/segment-anything.git
