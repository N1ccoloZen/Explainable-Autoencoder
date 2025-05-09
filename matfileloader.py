import scipy.io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dirName = sys.argv[1]

concepts_list = []

#extract each concept

cache = {}

for j, files in enumerate(os.listdir(dirName)):
    #print("File name:", files)

    if  files not in cache:
        filename = os.path.join(dirName, files)   
        mat = scipy.io.loadmat(filename)
                #print("Key features:", mat.keys())
        anno = mat['anno']    
        anno_struct = anno[0, 0]    

        #print(anno_struct.dtype.names)

        #print("Image name:", anno_struct['imname'])
        
    
        for i in range(anno_struct['objects'].shape[1]):
            #print("Object", i ,"name:", anno_struct['objects'][0,i][0][0])
            parts = anno_struct['objects'][0,i][3]
                #print(parts.shape)
            for i in range(parts.shape[1]):
                part = parts[0, i]
                label = part[0][0]
                mask = part[1]
                    
                if mask.sum() > 0:
                    #print("This mask exists")
                    #print("Concept", label, "exists in photo:", anno_struct['imname'])

                    if label not in concepts_list:
                        concepts_list.append(label)

print(j)
print("Size:", len(concepts_list))
#print(concepts_list)

#create a dataframe with this dataset

data_rows = []
concepts_set = sorted(set(concepts_list))

#print("Concepts set:", concepts_set)

#print(concepts_set[3])
"""
new = ", ".join(map(str, concepts_set))
print("Concepts set:", new)

wheel_count = new.count('wheel_')
headlight_count = new.count('headlight_')
door_count = new.count('door_')
window_count = new.count('window_')
engine_count = new.count('engine_')
cback_count = new.count('cbackside_')
cfront_count = new.count('cfrontside_')
cleft_count = new.count('cleftside_')
cright_count = new.count('crightside_')
croof_count = new.count('croofside_')

print("Wheel:", wheel_count)
print("Headlight:", headlight_count)
print("Door:", door_count)
print("Window:", window_count)
print("engine:", engine_count)
print("Cback:", cback_count)
print("Cfront:", cfront_count)
print("Cleft:", cleft_count)
print("Cright:", cright_count)
print("Croof:", croof_count)


"""

concepts_list = sorted(list({str(c) for c in concepts_list}))

label_to_avoid = ['boat', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

for files in os.listdir(dirName):
    filename = os.path.join(dirName, files)
    mat = scipy.io.loadmat(filename)        
    anno = mat['anno']
    anno_struct = anno[0, 0]

    for i in range(anno_struct['objects'].shape[1]):
            #print("Object", i ,"name:", anno_struct['objects'][0,i][0][0])
        obj = anno_struct['objects'][0, i]
        label_name = anno_struct['objects'][0,i][0][0]
        if label_name in label_to_avoid:
            continue
        parts = anno_struct['objects'][0,i][3]

        row = [files, i, label_name] + [0] * len(concepts_list)

            #print(parts.shape)
        for j in range(parts.shape[1]):
            part = parts[0, j]
            label = part[0][0]
            mask = part[1]

            if label in concepts_set and mask.sum() > 0:
                    #print("This mask exists")
                    #print("Concept", label, "exists in photo:", anno_struct['imname'])
                #idx = concepts_list.index(label)
                #concepts_binary[idx] = 1
                row[3 + concepts_list.index(label)] = 1
            
        data_rows.append(row)

#concepts_list = sorted(list({str(c) for c in concepts_list}))

dataset = pd.DataFrame(data_rows, columns = ['ID', 'label', 'obj_id'] + concepts_list)
dataset = dataset.drop(columns=['screen', 'pot', 'plant', 'cap', 'label'])
dataset = dataset.fillna(0)
dataset = dataset.rename(columns={'obj_id': 'label'})
#image_df = dataset[dataset['ID'] == '2008_000009.mat']
#print(image_df.iloc[0].drop(['ID', 'label']).sort_values(ascending=False).head(20))

dataset.to_csv('Pascal10Prova1.csv', index=False)
