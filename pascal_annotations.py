import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import nn 
#from torch.nn import Conv2d, MaxPool2d, Linear
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split #delete
from torchvision.transforms import Lambda
#from torchview import draw_graph
#from torchinfo import summary
#import hiddenlayer as hl
from torchvision import models
from timeit import default_timer as timer
#import warnings
import random
import copy
import os
import re
import sys

''' 

    args: 
        model: model to train
        learning_rate: learning rate 
        num_epochs: number of epochs
        batch_size = batch size
        decay: weight decay (L2 regularization)
        optimizer: optimizer to use
        criterion: loss to use
        early_stopping: early stopping epochs
        tol: tolerance for early stopping
        mom: momentum rate
        nest_mom: Nesterov's momentum

'''

''' 
    Obtain 1 row per image, each rows is selected as follows:
        1. An image can contain multiple objects and so multiple concepts can be activated
        2. For each image, we select the most frequent label or object
        3. We keep the row that activates the most concepts
'''

df = pd.read_csv('Pascal10Concepts_filtered.csv')

concept_col = [col for col in df.columns if col not in ['ID', 'label']]

filter_rows = []

for img_id, group_by_ID in df.groupby('ID'):

    label_counts = group_by_ID['label'].value_counts()
    most_common_label = label_counts.idxmax() #this is most freq label

    label_group = group_by_ID[group_by_ID['label'] == most_common_label].copy()
    label_group['sum_of_concepts'] = label_group[concept_col].sum(axis=1) #look for the one with most concepts activated

    most_concepts_active = label_group.loc[label_group['sum_of_concepts'].idxmax()].drop('sum_of_concepts') #this is the one with most concepts activated

    filter_rows.append(most_concepts_active)

filtered_df = pd.DataFrame(filter_rows)
filtered_df.to_csv('Pascal10_1RowPerImage.csv', index=False)

print(filtered_df.shape)

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

print(filtered_df.shape)
print('Created col:', multi_concepts)
print('Dropped col:', col_to_drop)
print(filtered_df)
filtered_df.to_csv('Pascal10_1RowPerImage_Concepts_filtered.csv', index=False)

annotations_file = 'Pascal10_1RowPerImage_Concepts_filtered.csv'
images_dir = sys.argv[1] #/Users/niccolozenaro/Università/Machine Learning/VOCdevkit/VOC2010/JPEGImages

class CustomImgSegmentationsDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(annotations_file)
        df['img_base'] = df.iloc[:, 0].apply(lambda x: os.path.splitext(x)[0])
        df['img_path'] = df['img_base'].apply(lambda x: os.path.join(img_dir, x  + '.jpg'))

        self.img_data = df[df['img_path'].apply(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):

        row = self.img_data.iloc[index]
        img_path = row['img_path']
        label = row.iloc[1]
        img_id = row['img_base']
        
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label, img_id
    

dataset = CustomImgSegmentationsDataset(images_dir, annotations_file, transform=None)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.ToTensor()
    ])
other_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data, test_data, val_data = random_split(dataset, [0.7, 0.2, 0.1])
print(f"Dataset shape after splitting: training={len(train_data)}, testing={len(test_data)}, validation={len(val_data)}")

train_data.dataset.transform = train_transform
val_data.dataset.transform = other_transform
test_data.dataset.transform = other_transform

batch_size = 2
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

for img, label, img_id in train_loader:
    print(f"Image shape: {img.shape[0]}, label: {label[0]}, img_id: {img_id[0]}")

    image = img[0].numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.title(f"Label: {label[0]}, img_id: {img_id[0]}")
    plt.show()
    break
    