# Puzzle-Pieces-Picker
Puzzle Pieces Picker: Deciphering Ancient Chinese Characters with Radical Reconstruction 

[Demo](http://vlrlabmonkey.xyz:7684)  
[Dataset](https://figshare.com/s/c5eedcb5069c10a08830)

<!-- ## Radical_Decomposition
```bash
conda create --name Decomposition python=3.9
conda activate Decomposition
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
cd ./Radical_Decomposition/
pip install -r requirements.txt
cd ./sam-hq-main/ 
pip install -e .

```
```bash
cd sam
python sam.py
cd ..
```
```bash
cd opencv
python opencv.py
cd ..
```
```bash
python build_dataset.py
python cut.py
python mocodataset.py
python feature.py
python filter.py
```
## Radical_Reconstruction
```bash
conda create -n Reconstruction python=3.8 -y
conda activate Reconstruction
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
cd ./Radical_Reconstruction/
pip install -r requirements.txt
pip install nltk==3.8.1

```
```bash
cd Dataset_Generation
python Deciphering_dataset.py
python -u main.py | tee log/Deciphering_dataset/OBS_train.log

``` -->