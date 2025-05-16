import scipy.io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
'''
dirName = sys.argv[1] #directory where the img dataset is stored

concepts_list = []

#extract each concept

cache = {}

for j, files in enumerate(os.listdir(dirName)):

    if  files not in cache:
        filename = os.path.join(dirName, files)   
        mat = scipy.io.loadmat(filename)        
        anno = mat['anno']    
        anno_struct = anno[0, 0]    
    
        for i in range(anno_struct['objects'].shape[1]):
            parts = anno_struct['objects'][0,i][3]
            
            for i in range(parts.shape[1]):
                part = parts[0, i]
                label = part[0][0]
                mask = part[1]
                    
                if mask.sum() > 0 and label not in concepts_list:
                    concepts_list.append(label)

#create a dataframe with this dataset

data_rows = []
concepts_set = sorted(set(concepts_list))

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
        obj = anno_struct['objects'][0, i]
        label_name = anno_struct['objects'][0,i][0][0]
        if label_name in label_to_avoid:
            continue
        parts = anno_struct['objects'][0,i][3]

        row = [files, i, label_name] + [0] * len(concepts_list)

        for j in range(parts.shape[1]):
            part = parts[0, j]
            label = part[0][0]
            mask = part[1]

            if label in concepts_set and mask.sum() > 0:
                row[3 + concepts_list.index(label)] = 1
            
        data_rows.append(row)

dataset = pd.DataFrame(data_rows, columns = ['ID', 'label', 'obj_id'] + concepts_list)
dataset = dataset.drop(columns=['screen', 'pot', 'plant', 'cap', 'label'])
dataset = dataset.fillna(0)
df = dataset.rename(columns={'obj_id': 'label'})

#df.to_csv('/Users/niccolozenaro/Università/Machine Learning/Explainable AutoEncoder/csvFiles/AnnotationsFile.csv', index=False)
'''
''' 
    Obtain 1 row per image, each rows is selected as follows:
        1. An image can contain multiple objects and so multiple concepts can be activated
        2. For each image, we select the most frequent label or object
        3. We keep the row that activates the most concepts based on sum or percentage
'''

df = pd.read_csv('/Users/niccolozenaro/Università/Machine Learning/Explainable AutoEncoder/csvFiles/AnnotationsFile.csv')

concept_col = [col for col in df.columns if col not in ['ID', 'label']]

filter_rows = []
'''

Naive dictionary to count the percentages of concept per label
The other approach instead calculates the avg attivations of concepts per label

concept_per_label = {
    'aeroplane': 31,
    'bicycle': 7,
    'bird': 13,
    'bus': 49,
    'car': 31,
    'cat': 17,
    'cow': 19,
    'dog': 18,
    'horse': 21,
    'motorbike': 6,
    'person': 24,
    'sheep': 19,
    'train': 55 
}
'''

df['num_active_concepts'] = df[concept_col].sum(axis=1)
concept_per_label = (df.groupby('label')['num_active_concepts'].mean().to_dict())

perc = False  #set to true if you want to use percentage

def compute_concept_percentage(row):
    label =row['label']
    active_concepts = row[concept_col].sum()
    max_concepts = concept_per_label.get(label, 1)

    return active_concepts / max_concepts

for img_id, group_by_ID in df.groupby('ID'):

    label_counts = group_by_ID['label'].value_counts()
    most_freq = label_counts.max()
    #most_common_label = label_counts.idxmax() #this is most freq label

    most_common_label = label_counts[label_counts == most_freq].index.to_list() #all labels with hightest freq
    label_group = group_by_ID[group_by_ID['label'].isin(most_common_label)].copy() #most freq labels, we can have a tie so we count concepts
    
    if perc:
        label_group['percentage'] = label_group.apply(compute_concept_percentage, axis=1)
        row_most_concepts = label_group.loc[label_group['percentage'].idxmax()].drop('percentage') 
    else:
        label_group['sum_of_concepts'] = label_group[concept_col].sum(axis=1) #look for the one with most concepts activated
        row_most_concepts = label_group.loc[label_group['sum_of_concepts'].idxmax()].drop('sum_of_concepts') #this is the one with most concepts activated, resolve the tie
 
    filter_rows.append(row_most_concepts)

filtered_df = pd.DataFrame(filter_rows)
#filtered_df.to_csv('/Users/niccolozenaro/Università/Machine Learning/Explainable AutoEncoder/csvFiles/Pascal10_1RowPerImage.csv', index=False)

''' 
    Now the dataframe has to be processed a little bit more. It contains repetitions of concepts (eg. wheel appears 8 times, window 20 times).
    If an objects has more than 4 of those repetitive concepts, we sobtitute those concepts with a 'lots_of_X' or 'multi_X' concept.
'''

col_to_drop = set()
multi_concepts = set()

matching_name = re.compile(r"^(.*)_(\d+)$")

for col in concept_col:

    match = matching_name.match(col)

    if match:
        concept, idx = match.groups()
        idx = int(idx)
        if idx > 4:
            multi_column = f"multi_{concept}"
            if multi_column not in filtered_df.columns:
                filtered_df[multi_column] = 0
                multi_concepts.add(multi_column)
            filtered_df[multi_column] |= filtered_df[col] #bit-wise OR expression
            col_to_drop.add(col)

filtered_df.drop(columns=col_to_drop, inplace=True)

#print(filtered_df.shape)
#print('Created col:', multi_concepts)
#print('Dropped col:', col_to_drop)
print(filtered_df)
#filtered_df.to_csv('/Users/niccolozenaro/Università/Machine Learning/Explainable AutoEncoder/csvFiles/Pascal10Percentage.csv', index=False)

