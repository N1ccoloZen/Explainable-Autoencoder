import scipy.io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dirName = sys.argv[1]

concepts_list = []

#extract each concept

for i, files in enumerate(os.listdir(dirName)):
    print("File name:", files)

    #if  files == '2008_003497.mat':
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

print(i)
print("Size:", len(concepts_list))
print(concepts_list)

#create a dataframe with this dataset

data_rows = []
concepts_set = set(concepts_list)

label_to_avoid = ['boat', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

for files in os.listdir(dirName):
    filename = os.path.join(dirName, files)
    mat = scipy.io.loadmat(filename)        
    anno = mat['anno']
    anno_struct = anno[0, 0]

    for i in range(anno_struct['objects'].shape[1]):
            #print("Object", i ,"name:", anno_struct['objects'][0,i][0][0])
        label_name = anno_struct['objects'][0,i][0][0]
        if label_name in label_to_avoid:
            continue

        parts = anno_struct['objects'][0,i][3]

        row = [files, label_name] + [0] * len(concepts_list)

            #print(parts.shape)
        for j in range(parts.shape[1]):
            part = parts[0, j]
            label = part[0][0]
            mask = part[1]

            if label in concepts_set and mask.sum() > 0:
                    #print("This mask exists")
                    #print("Concept", label, "exists in photo:", anno_struct['imname'])
                row[concepts_list.index(label)+2] = 1
            
        data_rows.append(row)

concepts_list = list({str(c) for c in concepts_list})

dataset = pd.DataFrame(data_rows, columns = ['ID', 'label'] + concepts_list)
dataset = dataset.drop(columns=['screen', 'pot', 'plant', 'cap'])
dataset = dataset.fillna(0)
dataset.to_csv('Pascal10Concepts_filtered.csv', index=False)





