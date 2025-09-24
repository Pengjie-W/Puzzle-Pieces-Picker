import argparse
import copy
import json
import os
import shutil
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Filter and reorganize radical decomposition results.")
    parser.add_argument("--feature-bank", dest="feature_bank_path", default="./output/feature_bank.pth",
                        help="Path to the cached feature bank tensor.")
    parser.add_argument("--label-bank", dest="label_bank_path", default="./output/label_bank.json",
                        help="Path to the label bank JSON file.")
    parser.add_argument("--target-bank", dest="target_bank_path", default="./output/target_bank.json",
                        help="Path to the target bank JSON file containing image paths.")
    parser.add_argument("--test-set", dest="test_set_path",
                        default="../Radical_Reconstruction/Dataset_Generation/test.json",
                        help="Path to the JSON file that lists test-set hanzi.")
    parser.add_argument("--radical-weight", dest="radical_weight_path", default="./data/radical_weight.json",
                        help="Path to the radical weight JSON database.")
    parser.add_argument("--hanzi", dest="hanzi_path",
                        default="../Radical_Reconstruction/Dataset_Generation/hanzi.json",
                        help="Path to the hanzi-to-radical JSON database.")
    parser.add_argument("--save-path", dest="save_path", default="./output/results_train",
                        help="Directory where filtered images will be copied.")
    return parser.parse_args()


def load_resources(args):
    feature_bank = torch.load(args.feature_bank_path)
    with open(args.label_bank_path, "r", encoding="utf8") as f:
        label_bank = json.load(f)
    with open(args.target_bank_path, "r", encoding="utf8") as f:
        target_bank = json.load(f)
    with open(args.test_set_path, "r", encoding="utf8") as f:
        test = json.load(f)

    feature_bank = feature_bank.cuda()

    with open(args.radical_weight_path, "r", encoding="utf8") as f:
        weight = json.load(f)
    weight[''] = weight['𢦏']

    with open(args.hanzi_path, "r", encoding="utf-8") as file:
        hanzis = json.load(file)
    hanzis[''] = hanzis['𢦏']

    return feature_bank, label_bank, target_bank, test, weight, hanzis

def process_folder(label_bank, target_bank, weight):
    """
    Build a label → file-list mapping and reassign per-image radical weights
    based on folder size (i.e., the number of images for that label).

    Args:
        label_bank (list): List of labels for all images.
        target_bank (list): List of absolute/relative image file paths.
        weight (dict): Base radical-weight lookup, keyed by hanzi.

    Returns:
        tuple:
            - label_to_folder (dict): Maps a normalized label to a list of
              image paths that belong to that label.
            - weight_new (dict): Per-image radical-weight dictionaries after
              redistributing weights by the folder size.
    """
    """
    得到label到文件夹的路径，并按照文件夹中文件的大小重新分配权值
    
    参数：
        label_bank (list): 标签列表
        target_bank (list): 所有图像的文件路径列表
        weight_dict (dict): 包含不同类别权重值的字典
    
    返回：
        tuple: 
            - label_to_folder (dict): 标签到对应文件夹，每个文件夹就是一个列表包含该标签的所有图像路径
            - weight_new (dict): 包含按照文件夹中数量，每张图像权重分配的字典
    """
    label_to_folder = {}
    weight_new = {}

    for i in range(len(label_bank)):
        # Normalize path and extract filename
        path = target_bank[i]
        filename = os.path.basename(path)
        if filename.replace('.jpg', '') == '0':
            label = label_bank[i]
            weight_new[path] = {label[2]: 1}
            continue

        label = label_bank[i]
        # Prefix label by data source (OpenCV vs SAM) for disambiguation
        if 'opencv' in path:
            label = 'O' + label
        else:
            label = 'S' + label

        # Group image paths by normalized label
        if label not in label_to_folder:
            label_to_folder[label] = [path]
        else:
            label_to_folder[label].append(path)

    # For each folder, divide base radical weights equally across its images
    for key, value in label_to_folder.items():
        divisor = len(value)  # number of images for this label (excl. undivided 0)
        for i in value:
            temp = copy.deepcopy(weight[key[3]])
            for k in temp:
                temp[k] /= divisor
            weight_new[i] = temp
    return label_to_folder, weight_new

def get_knn(weight_new, feature_bank, target_bank, label_bank, test):
    """
    Aggregate per-image radical distributions from the K-nearest neighbors
    in feature space.

    For each image:
      1) Find its similarities to all features.
      2) Apply a soft weighting (exp(sim / 0.1)) normalized by the top score.
      3) Collect up to k neighbors with distinct hanzi labels (or singleton
         weight entries), skipping test-set hanzi.
      4) Linearly combine the neighbors' radical distributions with similarity
         weights to get the final distribution.

    Args:
        weight_new (dict): Maps image path → {radical: weight}.

    Returns:
        dict: Maps image path → aggregated radical distribution (dict).
    """
    """
    计算基于特征相似度的K近邻权重分布
    
    参数:
        weight_new (dict): 包含样本路径到权重部首映射的字典
        
    返回:
        knn: 包含每个样本基于K近邻的加权部首权重分布
    """
    knn={}
    for ii in tqdm(range(feature_bank.size(1))):
        basepath = target_bank[ii]
        if basepath not in weight_new:
            continue
        baselabel = label_bank[ii] 
        baselabel = baselabel[2] # Chinese character label
        if baselabel in test: 
            continue
        filename = os.path.basename(basepath)
        if filename.replace('.jpg', '') == '0': 
            path = target_bank[ii]
            knn[path] = weight_new[path]
            continue
        temp = feature_bank.permute(1, 0)[ii]
        temp = temp.unsqueeze(0) 
        sim_matrix = torch.mm(temp, feature_bank) 
        sim_weight, sim_indices = sim_matrix.topk(k=feature_bank.size(1), dim=-1)
        sim_weight = (sim_weight / 0.1).exp()
        sim_weight = sim_weight / float(sim_weight[0][0]) # Normalized weights
        labels = [] # distinct hanzi labels for top-k
        weights = [] # corresponding similarity weights
        weightsdata = [] # corresponding radical distributions
        num = 0 # 
        for nnn in range(feature_bank.size(1)): # Iterate over all features
            if num >= 10:
                break
            index = int(sim_indices[0][nnn]) 
            path = target_bank[index] 
            label = label_bank[index] 
            label = label[2] # Chinese character label
            if label in test: 
                continue
            if path in weight_new: 
                if label not in labels or len(weight_new[path]) == 1: # Check if the label already exists in labels
                    labels.append(label) 
                    weights.append(float(sim_weight[0][nnn])) 
                    weightsdata.append(weight_new[path]) 
                    num += 1 
        if basepath in weight_new: 
            data = copy.deepcopy(weight_new[basepath]) 
        for key, value in data.items(): 
            data[key] = 0 # Initialize the weights to 0
        for i in range(len(labels)):
            weight = weights[i]
            weightsdatatemp = weightsdata[i]
            if i == 0:
                for key, value in weightsdatatemp.items():
                    if key in data:
                        data[key] = weight * value + data[key] # Update weights
            else:
                for key, value in weightsdatatemp.items():
                    if key in data:
                        data[key] = weight *value + data[key] # Update weights

        path = target_bank[ii] 
        knn[path] = data 
    return knn

def get_correct(knn, label_to_folder, test, hanzis):
    """
    Check whether predicted radical sets match the ground-truth radical
    decomposition (ignoring structural operators like ⿰, ⿱, etc.).

    For each hanzi folder:
      - Predict the top radical for every image using its KNN distribution.
      - Remove structural operators from the ground-truth decomposition.
      - Compare multiset counts (Counter) of predicted radicals with each
        operator-free decomposition; mark as correct if any matches.

    Args:
        knn (dict): Image path → predicted radical distribution (dict).
        label_to_folder (dict): Label → list of image paths.

    Returns:
        list: Labels (hanzi folders) whose predicted radical multiset matches
              one of the valid decompositions.
    """
    """
    验证汉字结构预测的正确性
    
    参数：
    knn : dict
        KNN模型预测结果，键为图像的路径，值为部首预测概率字典
    label_to_folder : dict
        label对应文件夹的图像路径字典
    
    返回：
    correct
        部首预测正确的汉字label列表
    """
    from collections import Counter
    correct=[] # Store the correct Chinese characters
    for key,value in label_to_folder.items(): # Traverse all folders, key is label, value is all image paths in the folder
        hanzi=key[3] # Chinese characters
        if hanzi in test: 
            continue
        predicted_radicals=[] 
        for i in value: # Iterate over all images in a folder
            radical_dist=knn[i] 
            predicted_radical = max(radical_dist, key=radical_dist.get)  
            predicted_radicals.append(predicted_radical) 
        
        radical=hanzis[key[3]]
        radical_new=[] # Store radical sequence removal structure
        for i in radical:
            radical_temp=[]
            for j in i:
                if j not in '⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻':
                    radical_temp.append(j)
            radical_new.append(radical_temp) 

        is_correct = False
        for i in radical_new:
            counter1 = Counter(predicted_radicals) 
            counter2 = Counter(i)
            if counter2==counter1:
                is_correct=True
                break
        if is_correct:
            correct.append(key)
    return correct


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def pic(corrects, label_to_folder, knn, save_path):
    """
    Copy correctly predicted images into folders named by their top radical.
    Before copying, apply a softmax over each image's radical distribution
    to ensure a proper probability simplex.

    Args:
        corrects (list): Labels (folders) deemed correct by `get_correct`.
        label_to_folder (dict): Label → list of image paths.
        knn (dict): Image path → radical distribution.
        save_path (str): Root directory to store reorganized images.

    Returns:
        dict: destination_path → top-radical probability (clipped to [0, 1]).
    """
    dataset = {}
    for i in corrects:
        hanzi = i[3]
        picpaths=label_to_folder[i] # Get all image paths in a folder
        for j in picpaths: 
            radical_dist = knn[j] 
            values = np.array(list(radical_dist.values()))
            softmax_values = softmax(values)
            for i, key in enumerate(radical_dist.keys()):
                radical_dist[key] = softmax_values[i]

            radical = max(radical_dist, key=radical_dist.get)
            folder_path=os.path.join(save_path,radical)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            source_path=j
            file_name = os.path.basename(source_path)
            folder = os.path.dirname(j)
            _, last_folder = os.path.split(folder)
            destination_path = os.path.join(folder_path,last_folder+'_'+file_name)
            shutil.copy(source_path, destination_path)
            if radical_dist[radical]<1:
                dataset[destination_path] =radical_dist[radical]
            else:
                dataset[destination_path]=1
                print(source_path,radical_dist[radical])
    return dataset


def main():
    args = parse_args()
    feature_bank, label_bank, target_bank, test, weight, hanzis = load_resources(args)

    label_to_folder, weight_new = process_folder(
        label_bank=label_bank,
        target_bank=target_bank,
        weight=weight,
    )
    knn = get_knn(
        weight_new=weight_new,
        feature_bank=feature_bank,
        target_bank=target_bank,
        label_bank=label_bank,
        test=test,
    )
    corrects = get_correct(
        knn=knn,
        label_to_folder=label_to_folder,
        test=test,
        hanzis=hanzis,
    )
    print(corrects)
    dataset = pic(
        corrects=corrects,
        label_to_folder=label_to_folder,
        knn=knn,
        save_path=args.save_path,
    )
    return dataset


if __name__ == "__main__":
    main()
