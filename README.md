# Puzzle-Pieces-Picker

**Puzzle Pieces Picker: Deciphering Ancient Chinese Characters with Radical Reconstruction**

ICDAR 2024 Oral
- 📄 [Paper](https://arxiv.org/abs/2406.03019)
- 🧪 [Demo](http://vlrlabmonkey.xyz:7684)
- 📦 [Dataset](https://huggingface.co/datasets/wpj20000/P3)

## Environment Setup
```bash
conda create --name P3 python=3.9
conda activate P3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
cd ./Radical_Decomposition/sam-hq-main/  # cloned from https://github.com/SysCV/sam-hq.git
pip install -e .
```

## Project Layout
- `Puzzle-Pieces-Picker/`
  - `Radical_Decomposition/`
    - `data/` OBI.zip — [Oracle bone inscription dataset](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Decomposition/data)
    - `sam/` sam_hq_vit_h.pth — [sam_hq_vit_h weights](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Decomposition/sam)(cloned from https://github.com/SysCV/sam-hq.git)
    - `moco/` model_last.pth — [Pretrained MoCo weights](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Decomposition/moco)
    - `output/` output.zip — [Segmentation outputs](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Decomposition/output)
  - `Radical_Reconstruction/`
    - `data/` ACCP.zip — [ACCP dataset](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Reconstruction/data)

You can download it to a folder and then unzip it
```bash
huggingface-cli download wpj20000/P3 \
  --repo-type dataset \
  --local-dir . \
  --local-dir-use-symlinks False \
  --resume \
  --max-workers 1

unzip Radical_Decomposition/data/OBI.zip
unzip Radical_Decomposition/output/output.zip
unzip Radical_Reconstruction/data/ACCP.zip

```
## Radical Decomposition Pipeline
These commands produce radical-level segments and features (or you can download the prepared `output` directory and move on to radical reconstruction).

```bash
cd ./Radical_Decomposition/opencv
python opencv.py        # contour-based segmentation with OpenCV
cd ../..
```

```bash
cd ./Radical_Decomposition/sam
python sam.py           # SAM-based segmentation
cd ../..
```

```bash
cd ./Radical_Decomposition
python build_dataset.py   # build the training dataset
python cut.py             # trim empty backgrounds
python mocodataset.py     # train MoCo (weights also provided in Radical_Decomposition/moco)
python feature.py         # extract features (feature vectors are saved under output)
python filter.py          # filter valid samples based on learned weights
```

## Radical Reconstruction Pipeline

```bash
cd ./Radical_Reconstruction/Dataset_Generation
python Deciphering_dataset.py             # build training/test sets for different historical periods
python Deciphering_dataset_OBIs_more.py   # add radical data from Radical_Decomposition
cd ..
mkdir -p log/Deciphering_dataset
```


```bash
# Train the oracle-bone-period model
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset OBS_train --test_dataset OBS_test \
  --device 0 --output_folder ./output/Deciphering_dataset/OBS_train/ \
  | tee log/Deciphering_dataset/OBS_train.log
```
You can either retrain the model or download and use the checkpoint for evaluation: [Checkpoint](https://huggingface.co/datasets/wpj20000/P3/tree/main/Radical_Reconstruction/output)

```bash
# Evaluate the oracle-bone-period model
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset OBS_train --test_dataset OBS_test \
  --device 0 --eval --resume "./output/Deciphering_dataset/OBS_train/checkpoints/checkpoint_ep0099.pth" \
  --output_folder ./output/Deciphering_dataset/OBS_train/ \
  | tee log/Deciphering_dataset/OBS_test.log
```

```bash
# Train the bronze-period model
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset Bronze_train --test_dataset Bronze_test \
  --device 0 --output_folder ./output/Deciphering_dataset/Bronze_train/ \
  | tee log/Deciphering_dataset/Bronze_train.log
```

```bash
# Evaluate the bronze-period model
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset Bronze_train --test_dataset Bronze_test \
  --device 0 --eval --resume "./output/Deciphering_dataset/Bronze_train/checkpoints/checkpoint_ep0099.pth" \
  --output_folder ./output/Deciphering_dataset/Bronze_train/ \
  | tee log/Deciphering_dataset/Bronze_test.log
```

```bash
# Train the Warring States-period model
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset Warring_train --test_dataset Warring_test \
  --device 0 --output_folder ./output/Deciphering_dataset/Warring_train/ \
  | tee log/Deciphering_dataset/Warring_train.log
```

```bash
# Evaluate the Warring States-period model
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset Warring_train --test_dataset Warring_test \
  --device 0 --eval --resume "./output/Deciphering_dataset/Warring_train/checkpoints/checkpoint_ep0099.pth" \
  --output_folder ./output/Deciphering_dataset/Warring_train/ \
  | tee log/Deciphering_dataset/Warring_test.log
```

```bash
# Train the oracle-bone* model (augmented with Radical_Decomposition results)
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset OBS_train_results_train --test_dataset OBS_test \
  --device 0 --output_folder ./output/Deciphering_dataset/OBS_train_results_train/ \
  | tee log/Deciphering_dataset/OBS_train_results_train.log
```

```bash
# Evaluate the oracle-bone* model
python -u main.py \
  --batch_size 256 --num_workers 4 \
  --train_dataset OBS_train_results_train --test_dataset OBS_test \
  --device 0 --eval --resume "./output/Deciphering_dataset/OBS_train_results_train/checkpoints/checkpoint_ep0099.pth" \
  --output_folder ./output/Deciphering_dataset/OBS_train_results_train/ \
  | tee log/Deciphering_dataset/OBS_test_results_train.log
```
···