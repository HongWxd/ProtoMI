# ProtoMI

Prototype-guided Molecular Inference (ProtoMI) is a graph-based molecular discovery framework for identifying promising electrolyte additive candidates under data-scarce conditions. The workflow combines unsupervised molecular representation learning, prototype contrastive learning, prototype-guided recommendation, and knowledge-guided post-screening.

## Overview

ProtoMI is designed for molecular recommendation when only a small number of experimentally reported positive molecules are available, while a much larger unlabeled candidate pool remains unexplored.

The pipeline contains four main stages:

1. **Data preprocessing**
   Convert reported additives and unlabeled candidate molecules from SMILES into graph data.

2. **Unsupervised representation learning (USL)**
   Train a graph neural network encoder on reported positive molecules using graph contrastive learning.

3. **Prototype contrastive learning (PCL)**
   Derive molecular prototypes from positive additives and adapt them to the unlabeled candidate space.

4. **Recommendation and post-screening**
   Recommend candidate molecules by prototype similarity and filter them using practical chemical constraints.

## Repository Structure

```text
ProtoMI/
├── data_preprocess/
│   ├── raw_data/                         # Raw literature / DOI data
│   ├── processed_data/                   # Processed additive-related files
│   ├── 1_labelling.py                    # Literature labelling
│   ├── 2_paper_classification.py         # Paper classification
│   ├── 3_summarize_by_CEI_SEI.py         # Functional summarization
│   ├── 4_reunify_additive_table.py       # Additive table unification
│   ├── 5_searching_space_collection.py   # Searching-space collection
│   └── download_papers.py
├── utils/
│   ├── data_loader.py                    # Dataset construction and train/test split
│   ├── USL.py                            # Unsupervised representation learning
│   ├── PCL.py                            # Prototype contrastive learning
│   ├── recommends.py                     # Prototype-guided recommendation
│   ├── post_screening.py                 # Knowledge-guided molecular filtering
│   ├── graph_augmentation.py             # Graph augmentation utilities
│   ├── tools.py                          # Molecular graph and utility functions
│   └── visualization.py
├── model.py                              # GNN encoders and projection heads
├── data_preprocess.py                    # Main preprocessing script
├── run_main.py                           # Main training / recommendation entry
├── run.sh                                # Example running script
└── README.md
```

## Installation

Install the required packages:

```bash
# PyTorch: please choose the version matching your CUDA environment
pip install torch torchvision torchaudio

# PyTorch Geometric
pip install torch-geometric

# Common scientific and chemistry packages
conda install -c conda-forge rdkit -y
pip install numpy pandas scipy scikit-learn matplotlib tqdm umap-learn requests openpyxl
```

If `torch-geometric` installation fails, please install the version that matches your local PyTorch and CUDA configuration following the official PyTorch Geometric installation instructions.

## Data Preparation

Before training, prepare a `data/` folder under the repository root:

```text
ProtoMI/
└── data/
    ├── additives_year.json # reported molecules
    └── searching_space_data_V2.csv # searching space
```

### 1. Positive additive file

`additives_year.json` should contain reported positive additive molecules. Each molecule should include at least a SMILES string and publication year.

Example format:

```json
{
  "LiDFOB": {
    "smiles": "O=C1O[B-](F)(F)OC1=O.[Li+]",
    "year": 2006
  },
  "TPFPB": {
    "smiles": "example_smiles",
    "year": 2015
  }
}
```

### 2. Searching-space file

`searching_space_data_V2.csv` should contain unlabeled candidate molecules. The current data loader expects the following columns:

```text
cid, formula, SMILES, fingerprint, topological, weight, heavy_atom
```

The `cid` column is used as the molecule ID, and the `SMILES` column is used to construct molecular graphs.

### 3. Run preprocessing

After preparing the two input files, run:

```bash
python data_preprocess.py
```

This script will generate the files required by the training pipeline:

```text
data/
├── additives_year_sorted.json
├── additive_id_mapping.csv
├── year_split_mapping.json
└── all_data_year.pkl
```

`all_data_year.pkl` stores the processed graph data. Its order is:

```text
[year-sorted positive additives] + [unlabeled candidate molecules]
```

This order is important because the year-split setting uses the positive molecule indices defined in `year_split_mapping.json`.

## Quick Start

The easiest way to run the full ProtoMI pipeline is:

```bash
bash run.sh
```

Before running, modify the following variables in `run.sh` according to your local environment:

```bash
save_path=./test
data_path=./data/all_data_year.pkl
additive_json_path=./data/additives_year_sorted.json
searching_space_path=./data/searching_space_data_V2.csv
post_screening_output_path=./test
device=cuda:0
split_year=all
```

If you do not have a GPU, set:

```bash
device=cpu
```

## Full Model Training

To train the full ProtoMI model directly using `run_main.py`, run:

```bash
python run_main.py \
  --method full_model \
  --random_state 42 \
  --data_path ./data/all_data_year.pkl \
  --save_path ./test \
  --searching_space_path ./data/searching_space_data_V2.csv \
  --additive_json_path ./data/additives_year_sorted.json \
  --year_mapping_path ./data/year_split_mapping.json \
  --post_screening_output_path ./test \
  --device cuda:0 \
  --usl_trials 10 \
  --pcl_trials 10 \
  --epoch 200 \
  --proto_epoch 300 \
  --EMA True \
  --use_decor_loss True \
  --use_topk True \
  --topk 25 \
  --encoder_similarity max \
  --usl_backbone GINE \
  --save_proto_drift False \
  --split_year all
```

The full model performs:

1. data loading;
2. USL encoder training or loading;
3. prototype extraction from positive additives;
4. PCL training or loading;
5. prototype-guided molecular recommendation;
6. post-screening of recommended molecules.

## Important Arguments

### Basic arguments

| Argument         | Description                                              | Default                     |
| ---------------- | -------------------------------------------------------- | --------------------------- |
| `--method`       | Recommendation method. Use `full_model` for ProtoMI.     | `full_model`                |
| `--data_path`    | Path to processed graph data.                            | `./data/all_data_year.pkl`  |
| `--save_path`    | Directory for saving checkpoints and intermediate files. | `checkpoints_origin_backup` |
| `--device`       | Device for training, e.g., `cuda:0` or `cpu`.            | `cuda:7` if available       |
| `--random_state` | Random seed.                                             | `42`                        |
| `--split_year`   | Year-split setting. Use `all` for all positives.         | `all`                       |

### USL arguments

| Argument                | Description                                                        | Default  |
| ----------------------- | ------------------------------------------------------------------ | -------- |
| `--usl_backbone`        | Backbone GNN for USL. Options include `GINE`, `GAT`, `GCN`, `GIN`. | `GINE`   |
| `--usl_trials`          | Number of USL trials.                                              | `10`     |
| `--epoch`               | Number of USL training epochs.                                     | `200`    |
| `--usl_batch_size`      | Batch size for USL training.                                       | `256`    |
| `--usl_learning_rate`   | Learning rate for USL.                                             | `0.0005` |
| `--usl_hidden_channels` | Hidden dimension of USL encoder.                                   | `256`    |
| `--dropout`             | Dropout rate.                                                      | `0.5`    |

### PCL arguments

| Argument              | Description                                                                  | Default   |
| --------------------- | ---------------------------------------------------------------------------- | --------- |
| `--pcl_trials`        | Number of PCL trials.                                                        | `10`      |
| `--proto_epoch`       | Number of PCL training epochs.                                               | `300`     |
| `--pcl_batch_size`    | Batch size for PCL training.                                                 | `1024`    |
| `--pcl_learning_rate` | Learning rate for PCL.                                                       | `0.00001` |
| `--temperature`       | Temperature coefficient for prototype contrastive learning.                  | `0.1`     |
| `--topk`              | Number of top candidate molecules selected for each prototype in each batch. | `25`      |
| `--EMA`               | Whether to use exponential moving average for prototype updating.            | `True`    |
| `--use_decor_loss`    | Whether to use prototype decorrelation regularization.                       | `True`    |
| `--use_topk`          | Whether to use top-k prototype-guided sample selection.                      | `True`    |
| `--save_proto_drift`  | Whether to save prototype centroids at each epoch.                           | `False`   |

### Post-screening arguments

| Argument                       | Description                                                  | Default                              |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------ |
| `--searching_space_path`       | Candidate molecule CSV file.                                 | `./data/searching_space_data_V2.csv` |
| `--additive_json_path`         | Year-sorted positive additive JSON file.                     | `./data/additives_year_sorted.json`  |
| `--post_screening_output_path` | Directory for saving post-screened molecules and embeddings. | `./outputs/`                         |
| `--save_molecules`             | Whether to save post-screened molecules.                     | `True`                               |

## Output Files

After running the full model, checkpoints and results will be saved to `save_path` and `post_screening_output_path`.

### Model checkpoints

```text
<save_path>/
├── USL_encoder_<epoch>_<usl_backbone>_<split_year>.pth
├── PCL_encoder_full_model_ema_<EMA>_decor_<use_decor_loss>_topk_<use_topk>_year_<split_year>.pth
├── PCL_projection_full_model_ema_<EMA>_decor_<use_decor_loss>_topk_<use_topk>_year_<split_year>.pth
└── proto_centroids_full_model_ema_<EMA>_decor_<use_decor_loss>_topk_<use_topk>_year_<split_year>.pth
```

### Prototype tables

During PCL training, prototype assignment tables are saved as:

```text
<save_path>/proto_table_trial_<trial>.csv
```

Each table contains positive molecule IDs and their assigned prototype labels.

### Recommendation results

After post-screening, the final recommended molecules are saved as:

```text
<post_screening_output_path>/
├── recommendations_after_post_screening_full_model_ema_True_decor_True_topk_True_year_all.csv
└── embeddings_after_post_screening_full_model_ema_True_decor_True_topk_True_year_all.npy
```

The CSV file contains the final post-screened molecule information, and the NPY file contains the corresponding molecular embeddings.

## Year-Split Evaluation

ProtoMI supports year-split experiments for temporal validation. For example:

```bash
python run_main.py \
  --method full_model \
  --data_path ./data/all_data_year.pkl \
  --searching_space_path ./data/searching_space_data_V2.csv \
  --additive_json_path ./data/additives_year_sorted.json \
  --year_mapping_path ./data/year_split_mapping.json \
  --save_path ./test_2019 \
  --post_screening_output_path ./test_2019 \
  --device cuda:0 \
  --split_year 2019
```

Available year-split options depend on `year_split_mapping.json`. Common settings include:

```text
all
2017
2019
2021
2023
```

`split_year=all` uses all positive additives for training and recommendation. A specific year such as `2019` uses only additives reported up to the corresponding cutoff year as training positives.

## Baseline Methods

Besides the full ProtoMI model, `run_main.py` also supports several baseline methods.

### Random recommendation

```bash
python run_main.py \
  --method random \
  --data_path ./data/all_data_year.pkl \
  --searching_space_path ./data/searching_space_data_V2.csv \
  --additive_json_path ./data/additives_year_sorted.json \
  --year_mapping_path ./data/year_split_mapping.json \
  --save_path ./random_results \
  --post_screening_output_path ./random_results \
  --device cuda:0
```

### Morgan fingerprint baseline

```bash
python run_main.py \
  --method morgan \
  --data_path ./data/all_data_year.pkl \
  --searching_space_path ./data/searching_space_data_V2.csv \
  --additive_json_path ./data/additives_year_sorted.json \
  --year_mapping_path ./data/year_split_mapping.json \
  --save_path ./morgan_results \
  --post_screening_output_path ./morgan_results \
  --device cuda:0
```

### Encoder-only baselines

```bash
# USL encoder-only positive-similarity baseline
python run_main.py \
  --method usl_encoder_only \
  --data_path ./data/all_data_year.pkl \
  --searching_space_path ./data/searching_space_data_V2.csv \
  --additive_json_path ./data/additives_year_sorted.json \
  --year_mapping_path ./data/year_split_mapping.json \
  --save_path ./test \
  --post_screening_output_path ./usl_encoder_only_results \
  --device cuda:0 \
  --encoder_similarity max

# PCL encoder-only positive-similarity baseline
python run_main.py \
  --method pcl_encoder_only \
  --data_path ./data/all_data_year.pkl \
  --searching_space_path ./data/searching_space_data_V2.csv \
  --additive_json_path ./data/additives_year_sorted.json \
  --year_mapping_path ./data/year_split_mapping.json \
  --save_path ./test \
  --post_screening_output_path ./pcl_encoder_only_results \
  --device cuda:0 \
  --encoder_similarity max
```

### Encoder clustering baselines

```bash
# USL encoder clustering baseline
python run_main.py \
  --method usl_encoder_clustering \
  --data_path ./data/all_data_year.pkl \
  --searching_space_path ./data/searching_space_data_V2.csv \
  --additive_json_path ./data/additives_year_sorted.json \
  --year_mapping_path ./data/year_split_mapping.json \
  --save_path ./test \
  --post_screening_output_path ./usl_encoder_clustering_results \
  --device cuda:0 \
  --cluster_num 7

# PCL encoder clustering baseline
python run_main.py \
  --method pcl_encoder_clustering \
  --data_path ./data/all_data_year.pkl \
  --searching_space_path ./data/searching_space_data_V2.csv \
  --additive_json_path ./data/additives_year_sorted.json \
  --year_mapping_path ./data/year_split_mapping.json \
  --save_path ./test \
  --post_screening_output_path ./pcl_encoder_clustering_results \
  --device cuda:0 \
  --cluster_num 7
```

Note that PCL-based baselines require pretrained PCL checkpoints in `save_path`.

## Example `run.sh`

You can use the following script as a clean template:

```bash
#!/bin/bash

save_path=./test
data_path=./data/all_data_year.pkl
additive_json_path=./data/additives_year_sorted.json
searching_space_path=./data/searching_space_data_V2.csv
post_screening_output_path=./test

method=full_model
recommend_model=full_model
seed=42

usl_trials=10
epoch=200
pcl_trials=10
proto_epoch=300

EMA=True
use_decor_loss=True
use_topk=True
device=cuda:0

encoder_similarity=max
usl_backbone=GINE
save_proto_drift=False
split_year=all

echo "======================================"
echo "Running ProtoMI recommendation pipeline"
echo "METHOD: $method"
echo "SEED: $seed"
echo "USE_EMA: $EMA"
echo "USE_DECOR_LOSS: $use_decor_loss"
echo "USE_TOPK: $use_topk"
echo "DEVICE: $device"
echo "SPLIT_YEAR: $split_year"
echo "======================================"

python run_main.py \
  --method $method \
  --random_state $seed \
  --data_path $data_path \
  --save_path $save_path \
  --usl_trials $usl_trials \
  --pcl_trials $pcl_trials \
  --epoch $epoch \
  --proto_epoch $proto_epoch \
  --EMA $EMA \
  --use_decor_loss $use_decor_loss \
  --use_topk $use_topk \
  --device $device \
  --searching_space_path $searching_space_path \
  --additive_json_path $additive_json_path \
  --recommend_model $recommend_model \
  --encoder_similarity $encoder_similarity \
  --usl_backbone $usl_backbone \
  --save_proto_drift $save_proto_drift \
  --split_year $split_year \
  --post_screening_output_path $post_screening_output_path
```

Run it with:

```bash
bash run.sh
```

## Notes

1. The post-screening step queries PubChem synonyms to check whether a molecule has a CAS-like identifier. This step requires internet access and may take time for large candidate sets.

2. If post-screening fails because of temporary PubChem connection errors, rerun the pipeline or manually inspect the saved intermediate candidates.

3. If you want to rerun USL or PCL training from scratch, remove the corresponding `.pth` checkpoint files from `save_path`.

4. If CUDA memory is insufficient, reduce `--pcl_batch_size` and `--usl_batch_size`, or run on CPU for debugging.

## Citation

If you use this repository, please cite the corresponding ProtoMI paper:

```bibtex
@article{protomi,
  title   = {ProtoMI: Prototype-guided Molecular Inference for Data-scarce Electrolyte Additive Discovery},
  author  = {Hong, Weixiang and co-authors},
  journal = {To be updated},
  year    = {2026}
}
```
