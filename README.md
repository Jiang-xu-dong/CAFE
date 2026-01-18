# CAFE

This repository is the official implementation of "Cross-source Evidence Tells Us More: Spotting Metaphorical NSFW Content".

## 📦 Environment Setup

Please install the required dependencies using:

### Option 1: Create the environment from `environment.yaml` 

```bash
conda env create -f environment.yaml
conda activate CAFE
```

### Option 2: Create the environment manually with Conda and Pip
```bash
conda create -n CAFE python=3.9
conda activate CAFE
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training Procedure

### Steps:

```bash

python tools/train.py     # Train alignment process
python tools/rel_train.py # Train relation prediction
python tools/train_detect.py # Train NSFW detection
```

---


## Dataset

This dataset link will be updated upon acceptance

## 📦 Hardware

This project was developed and tested on an NVIDIA A100 GPU.

## 📁 File Structure Description

```
|-- README.MD
|-- blip-vqa-base     # Attribute extractor
|-- configs           # Configuration files
|   |-- mask2former   # Entity segmentation
|   `-- unitrack      # Entity track
|-- datasets          # Data processing files
|   |-- datasets      # Load data
|   `-- pipelines     # Process data
|-- environment.yaml  # Environment for project
|-- models            # Model files
|   |-- mask2former   # Entity segmentation
|   |-- relation_head # Relation prediction
|   `-- unitrack      # Entity track
|-- requirements.txt  # requirements for project
|-- src               # source files
|   |-- core          # fusion detector
|   |-- models        # model files for fusion 
|   `-- utils         # Load data
|-- tools             # Training code
|   |-- rel_train.py  # Train relation prediction
|   |-- train.py      # Train entity segmentation
|   `-- train_gnn.py  # Video scene graph detection training
`-- utils             # Utility scripts
    |-- relation_matching.py  # Match relation
    `-- ...
