import copy
import json
import os
dataset=[]
for root, directories, files in os.walk('./output/Decomposition'):
    for file in files:
        file_path = os.path.join(root, file)
        second_to_last_folder = os.path.basename(os.path.dirname(file_path))
        data={}
        data['path']=file_path
        data['label']=second_to_last_folder
        if '0.jpg' in files and len(files)==2:
            dataset.append(copy.deepcopy(data))
            break
        dataset.append(copy.deepcopy(data))
with open('./output/Decomposition_Dataset.json','w',encoding='utf8') as f:
    json.dump(dataset, f, ensure_ascii=False)
