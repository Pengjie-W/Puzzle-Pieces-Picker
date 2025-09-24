# Puzzle-Pieces-Picker

**Puzzle Pieces Picker: Deciphering Ancient Chinese Characters with Radical Reconstruction**

ICDAR 2024 Oral
- 📄 [Paper](https://arxiv.org/abs/2406.03019)
- 🧪 [Demo](http://vlrlabmonkey.xyz:7684)
- 📦 [Dataset](https://huggingface.co/datasets/wpj20000/P3)

## Quick Start

### 1. Environment setup
```bash
conda create --name P3 python=3.9
conda activate P3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
git clone https://github.com/Pengjie-W/Puzzle-Pieces-Picker.git
cd Puzzle-Pieces-Picker
pip install -r requirements.txt
cd ./Radical_Decomposition/sam-hq-main  # cloned from https://github.com/SysCV/sam-hq.git
pip install -e .
```

### 2. Repository layout
- `Puzzle-Pieces-Picker/`
  - `Radical_Decomposition/`
    - `data/` OBI.zip — [Oracle bone inscription dataset](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Decomposition/data)
    - `sam/` sam_hq_vit_h.pth — [sam_hq_vit_h weights](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Decomposition/sam)(cloned from https://github.com/SysCV/sam-hq.git)
    - `moco/` model_last.pth — [Pretrained MoCo weights](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Decomposition/moco)
    - `output/` output.zip — [Segmentation outputs](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Decomposition/output)
  - `Radical_Reconstruction/`
    - `data/` ACCP.zip — [ACCP dataset](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Reconstruction/data)
<!-- ```text
Puzzle-Pieces-Picker/
|-- Radical_Decomposition/
|   |-- data/  (OBI.zip -> Oracle bone inscription dataset)
|   |-- sam/   (sam_hq_vit_h.pth -> SAM-HQ weights)
|   |-- moco/  (model_last.pth -> trained MoCo weights)
|   |-- output/ (output.zip -> prepared segmentation outputs)
|-- Radical_Reconstruction/
    |-- data/  (ACCP.zip -> ACCP dataset)
``` -->
Download the archives from Hugging Face and unzip them in place:
```bash
cd ../.. # cd Puzzle-Pieces-Picker/
pip install -U "huggingface_hub[cli]"
huggingface-cli download wpj20000/P3 \
  --repo-type dataset \
  --local-dir . \
  --local-dir-use-symlinks False \
  --resume \
  --max-workers 1 \
  --exclude "README.md"

for archive in \
  Radical_Decomposition/data/OBI.zip \
  Radical_Decomposition/output/output.zip \
  Radical_Reconstruction/data/ACCP.zip
  do unzip "$archive" -d "$(dirname "$archive")/"; done
```

## Radical Decomposition pipeline
Produces radical-level segments and features. Run each step from the repository root.
(or you can download the prepared `output` directory and move on to radical reconstruction).

**1. Contour-based segmentation (OpenCV)**
```bash
pushd Radical_Decomposition/opencv
python opencv.py \
  --data_path ../data \
  --json_file ../data/OBI.json \
  --output_dir ../output/Decomposition/opencv
popd
```

**2. SAM-based segmentation**
```bash
pushd Radical_Decomposition/sam
python sam.py \
  --data_path ../data \
  --json_file ../data/OBI.json \
  --output_dir ../output/Decomposition/sam \
  --gpu 0
popd
```

**3. Dataset creation, feature extraction, and filtering**
```bash
pushd Radical_Decomposition
python build_dataset.py \
  --input_dir ./output/Decomposition \
  --output ./output/Decomposition_Dataset.json
python cut.py \
  --dataset ./output/Decomposition_Dataset.json \
  --threshold 160
python mocodataset.py  # Train MoCo (or skip and use moco/model_last.pth)
python feature.py \
  --resume moco/model_last.pth \
  --feature-bank-path ./output/feature_bank.pth \
  --target-bank-path ./output/target_bank.json \
  --label-bank-path ./output/label_bank.json
# Add --use-paper-dataset to reproduce the paper's released feature bank
# python feature.py \
#   --resume moco/model_last.pth \
#   --feature-bank-path ./output/feature_bank.pth \
#   --target-bank-path ./output/target_bank.json \
#   --label-bank-path ./output/label_bank.json \
#   --use-paper-dataset
python filter.py \
  --feature-bank ./output/feature_bank_old.pth \
  --label-bank ./output/label_bank_old.json \
  --target-bank ./output/target_bank_old.json \
  --test-set ../Radical_Reconstruction/Dataset_Generation/test.json \
  --radical-weight ./data/radical_weight.json \
  --hanzi ../Radical_Reconstruction/data/hanzi.json \
  --save-path ./output/results_train
popd
```

## Radical Reconstruction pipeline
Generates period-specific datasets and trains reconstruction models.

**1. Build datasets**
```bash
pushd Radical_Reconstruction/Dataset_Generation
python Deciphering_dataset.py \
  --hanzi ../data/hanzi.json \
  --source ../data/source.json \
  --id-to-chinese ../data/ID_to_Chinese.json \
  --obs-root ../data/Dataset \
  --dataset-root ../data/dataset \
  --font-root ../data/Font_Generation \
  --train train.json \
  --test test.json \
  --output-dir Deciphering_dataset

# add radical data from Radical_Decomposition
python Deciphering_dataset_OBIs_more.py \
  --train train.json \
  --test test.json \
  --hanzi ../data/hanzi.json \
  --source ../data/source.json \
  --obs-train ./Deciphering_dataset/OBS_train.json \
  --results-root ../../Radical_Decomposition/output/results_train \
  --output ./Deciphering_dataset/OBS_train_results_train.json
popd
mkdir -p Radical_Reconstruction/log/Deciphering_dataset
```
A pretrained checkpoint for each period is available on Hugging Face under `Radical_Reconstruction/output`.

You can either retrain the model or download and use the checkpoint for evaluation: [Checkpoint](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Reconstruction/output)

**2. Train or evaluate**
Use the templates below, substituting the dataset names from the table.

```bash
# Training template
pushd Radical_Reconstruction
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset ${TRAIN_DATASET} \
  --test_dataset ${TEST_DATASET} \
  --device 0 \
  --source ../data/source.json \
  --output_folder ./output/Deciphering_dataset/${TRAIN_DATASET}/ \
  | tee log/Deciphering_dataset/${TRAIN_DATASET}.log

# Evaluation template
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset ${TRAIN_DATASET} \
  --test_dataset ${TEST_DATASET} \
  --source ../data/source.json \
  --device 0 --eval --resume "${CHECKPOINT}" \
  --output_folder ./output/Deciphering_dataset/${TRAIN_DATASET}/ \
  | tee log/Deciphering_dataset/${TEST_DATASET}.log
```

| Period | TRAIN_DATASET | TEST_DATASET | CHECKPOINT (if evaluating) |
| --- | --- | --- | --- |
| Oracle bone | OBS_train | OBS_test | ./output/Deciphering_dataset/OBS_train/checkpoints/checkpoint_ep0099.pth |
| Bronze | Bronze_train | Bronze_test | ./output/Deciphering_dataset/Bronze_train/checkpoints/checkpoint_ep0099.pth |
| Warring States | Warring_train | Warring_test | ./output/Deciphering_dataset/Warring_train/checkpoints/checkpoint_ep0099.pth |
| Oracle bone + radicals | OBS_train_results_train | OBS_test | ./output/Deciphering_dataset/OBS_train_results_train/checkpoints/checkpoint_ep0099.pth |
| Seal | Seal_train | Seal_test | ./output/Deciphering_dataset/Seal_train/checkpoints/checkpoint_ep0099.pth |
| Clerical | Clerical_train | Clerical_test | ./output/Deciphering_dataset/Clerical_train/checkpoints/checkpoint_ep0099.pth |
| Kangxi | Kangxi_train | Kangxi_test | ./output/Deciphering_dataset/Kangxi_train/checkpoints/checkpoint_ep0099.pth |
| Regular | Regular_train | Regular_test | ./output/Deciphering_dataset/Regular_train/checkpoints/checkpoint_ep0099.pth |

## Citation
If you find this work helpful, please cite:
```bibtex
@inproceedings{wang2024puzzle,
  title={Puzzle pieces picker: Deciphering ancient chinese characters with radical reconstruction},
  author={Wang, Pengjie and Zhang, Kaile and Wang, Xinyu and Han, Shengwei and Liu, Yongge and Jin, Lianwen and Bai, Xiang and Liu, Yuliang},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={169--187},
  year={2024},
  organization={Springer}
}
```
