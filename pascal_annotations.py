import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import nn 
#from torch.nn import Conv2d, MaxPool2d, Linear
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split #delete
from torchvision.transforms import Lambda
#from torchview import draw_graph
from torchsummary import summary
#import hiddenlayer as hl
from torchvision import models
from timeit import default_timer as timer
#import warnings
import hiddenlayer as hl
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
print("You are using:", sys.platform)
print(f"{torch.__version__=}")
print("MPS support=", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

# Set seed for reproducibility
torch.manual_seed(19)

df = pd.read_csv('Pascal10Prova1.csv')

concept_col = [col for col in df.columns if col not in ['ID', 'label']]

filter_rows = []

for img_id, group_by_ID in df.groupby('ID'):

    label_counts = group_by_ID['label'].value_counts()
    most_freq = label_counts.max()
    #most_common_label = label_counts.idxmax() #this is most freq label

    most_common_label = label_counts[label_counts == most_freq].index.to_list() #all labels with hightest freq
    label_group = group_by_ID[group_by_ID['label'].isin(most_common_label)].copy() #most freq labels, we can have a tie so we count concepts
    #label_group = group_by_ID[group_by_ID['label'] == most_common_label].copy()
    label_group['sum_of_concepts'] = label_group[concept_col].sum(axis=1) #look for the one with most concepts activated
    row_most_concepts = label_group.loc[label_group['sum_of_concepts'].idxmax()].drop('sum_of_concepts') #this is the one with most concepts activated, resolve the tie
    #most_concepts_active = label_group.loc[label_group['sum_of_concepts'].idxmax()].drop('sum_of_concepts') #this is the one with most concepts activated

    filter_rows.append(row_most_concepts)

filtered_df = pd.DataFrame(filter_rows)
#filtered_df.to_csv('Pascal10_1RowPerImage.csv', index=False)

#print(filtered_df.shape)

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
#print(filtered_df)
#filtered_df.to_csv('Pascal10_1RowPerImage_Concepts_filtered.csv', index=False)

annotations_file = 'Pascal10_1RowPerImage_Concepts_filtered.csv'

ordered_labels = sorted(filtered_df['label'].unique())

""" enc = OneHotEncoder()
enc.fit(np.array(ordered_labels).reshape(-1, 1))

encoded_labels = enc.transform(np.array(filtered_df['label']).reshape(-1, 1)).toarray()

 """

images_dir = sys.argv[1] #/Users/niccolozenaro/Università/Machine Learning/VOCdevkit/VOC2010/JPEGImages

class CustomImgSegmentationsDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(annotations_file)
        df['img_base'] = df.iloc[:, 0].apply(lambda x: os.path.splitext(x.strip())[0])
        df['img_path'] = df['img_base'].apply(lambda x: os.path.join(img_dir, x  + '.jpg'))

        #print(df['img_path'].head(5))
        ordered_labels = sorted(df['label'].unique())

        enc = OneHotEncoder()
        enc.fit(np.array(ordered_labels).reshape(-1, 1))

        encoded_labels = enc.transform(np.array(df['label']).reshape(-1, 1)).toarray()

        df['label'] = list(encoded_labels)

        self.img_data = df[df['img_path'].apply(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):

        row = self.img_data.iloc[index]
        img_path = row['img_path']
        label = torch.tensor(row['label'], dtype=torch.float32)
        img_id = row['img_base']
        concepts = torch.tensor(row.iloc[2:-2].values.astype(np.float32))
        
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label, img_id, concepts
    

dataset = CustomImgSegmentationsDataset(images_dir, annotations_file, transform=None)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.ToTensor()
    ])
other_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print('Dataset shape before splitting is:', dataset.__len__())

train_data, test_data, val_data = random_split(dataset, [0.7, 0.2, 0.1])
print(f"Dataset shape after splitting: training={len(train_data)}, testing={len(test_data)}, validation={len(val_data)}")

train_data.dataset.transform = train_transform
val_data.dataset.transform = other_transform
test_data.dataset.transform = other_transform

batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

concepts_list = list(pd.read_csv('Pascal10_1RowPerImage_Concepts_filtered.csv').columns)[2:]
#print(f"Concepts: {concepts}")

for img, label, img_id, concepts in train_loader:

    idx = label[0].argmax()

    print(f"Image shape: {img[0].shape}, label: {ordered_labels[idx]}, img_id: {img_id[0]}")
    
    for i, concept in enumerate(concepts[0]):
        if concept == 1:
            print(f'Binary concept {concepts_list[i]} is activated')

    image = img[0].numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.title(f"Label: {ordered_labels[idx]}, img_id: {img_id[0]}")
    plt.show()
    break
    
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
#ResNet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
#ResNet18 = resnet18(weights=ResNet18_Weights.DEFAULT)

#summary(ResNet50, (3, 224, 224), device=device)
#summary(ResNet18, (3, 224, 224), device=device)

class TuneCNNAttributes(nn.Module):
    def __init__(self, model, num_concepts, model_weights, freeze_backbone=True):
        super(TuneCNNAttributes, self).__init__()

        self.resnet = model(weights = model_weights)

        if freeze_backbone:
            for name, params in self.resnet.named_parameters():
                if 'fc' in name:
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_concepts)

    def forward(self, x):
        return self.resnet(x)
    
#training function
def trainFineTune (model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    
    loss_train, loss_val = [], []
    acc_train, acc_val = [], []
    history1 = hl.History()
    canvas1 = hl.Canvas()

    for epoch in range(num_epochs):
        model.train()
        tot_acc_train, tot_count_train, n_train_batches, tot_loss_train = 0, 0, 0, 0

        for img, label, img_ID, concepts in train_loader:
            img = img.to(device)
            concepts = concepts.to(device)
            #attribute = attributes[img_ID-1].to(device)
            optimizer.zero_grad()
            prediction = model(img)
            loss = criterion(prediction, concepts)

            tot_loss_train += loss.item()
            loss.backward()
            optimizer.step()

            tot_acc_train += ((prediction > 0.5).float() == concepts).sum().item()
            tot_count_train += concepts.numel()
            n_train_batches += 1

        avg_loss_train = tot_loss_train / n_train_batches
        loss_train.append(avg_loss_train)
        accuracy_train = (tot_acc_train / tot_count_train) * 100
        acc_train.append(accuracy_train)

        tot_acc_val, tot_count_val, n_val_batches, tot_loss_val = 0, 0, 0, 0

        with torch.no_grad():
            model.eval()
            for img, label, img_ID, concepts in val_loader:
                img = img.to(device)
                concepts = concepts.to(device)
                #attribute = attributes[img_ID-1].to(device)
                prediction = model(img)
                loss = criterion(prediction, concepts)

                tot_loss_val += loss.item()

                tot_acc_val += ((prediction > 0.5).float() == concepts).sum().item()
                tot_count_val += concepts.numel()
                n_val_batches += 1
        
        avg_loss_val = tot_loss_val / n_val_batches
        loss_val.append(avg_loss_val)
        accuracy_val = (tot_acc_val / tot_count_val) * 100
        acc_val.append(accuracy_val)

        if epoch % 1 == 0:
            history1.log(epoch, train_loss = avg_loss_train, train_accuracy = accuracy_train, val_loss = avg_loss_val, val_accuracy = accuracy_val)
            with canvas1:
                canvas1.draw_plot([history1["train_loss"], history1["val_loss"]], labels=['Training Loss', 'Validation Loss'])
                canvas1.draw_plot([history1["train_accuracy"], history1["val_accuracy"]], labels=['Training Accuracy', 'Validation Accuracy'])

    return loss_train, acc_train, loss_val, acc_val

#if you want to plot again

def to_cpu(tensor): #only for Apple M1/M2/M3 have to move tensors to CPU
    #if device == 'mps':
    return [t.cpu().item() if torch.is_tensor(t) else t for t in tensor]
    #else:
     #   [t.item() if torch.is_tensor(t) else t for t in tensor]

def plot_learning_acc_loss(loss_train, acc_train, loss_val, acc_val, name):
    
    plt.figure(figsize=(10, 12))

    loss_train = to_cpu(loss_train)
    acc_train = to_cpu(acc_train)
    loss_val = to_cpu(loss_val)
    acc_val = to_cpu(acc_val)

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.plot(range(len(acc_train)), acc_train, label="Training Accuracy")
    plt.plot(range(len(acc_val)), acc_val, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.grid()
    plt.plot(range(len(loss_train)), loss_train, label="Training Loss")
    plt.plot(range(len(loss_val)), loss_val, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')

    #plt.savefig(name + '.png')
    plt.show()
    plt.close()


num_epochs = 5
lr = 1e-3

for _, _, _, concepts in train_loader:
    num_concepts = concepts[0].shape[0]

""" ResNetTuned = TuneCNNAttributes(model=resnet18, num_concepts=num_concepts, model_weights=ResNet18_Weights.DEFAULT, freeze_backbone=True).to(device)
pos_weight = torch.tensor([11.0]*num_concepts).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(ResNetTuned.parameters(), lr=lr, weight_decay=5e-4)

start = timer()

torch.mps.empty_cache()

loss_train, acc_train, loss_val, acc_val = trainFineTune(ResNetTuned, train_loader, val_loader, criterion, optimizer, num_epochs, device)

end = timer()

print(f"Training took {end-start:.2f} seconds")

torch.mps.empty_cache()

plot_learning_acc_loss(loss_train, acc_train, loss_val, acc_val, 'img1') """

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_val, expand_dim=[]):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_val)
        self.batchnorm = nn.ModuleList()
        self.activation = nn.LeakyReLU()

        if len(expand_dim) == 0:
            self.layers.append(nn.Linear(input_dim, num_classes))
        else:
            for layer_idx in range(len(expand_dim)):
                if layer_idx == 0:
                    self.layers.append(nn.Linear(input_dim, expand_dim[0]))
                else:
                    self.layers.append(nn.Linear(expand_dim[layer_idx-1], expand_dim[layer_idx]))
                
                self.batchnorm.append(nn.BatchNorm1d(expand_dim[layer_idx]))

            self.layers.append(nn.Linear(expand_dim[-1], num_classes))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if len(self.layers) == 1:
            return self.layers[0](x)
        else:
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x)
                x = self.batchnorm[i](x)
                x = self.activation(x)
                x = self.dropout(x)
            return self.layers[-1](x)
        
def trainMLP(model, train_loader, val_loader, criterion, optimizer, num_epochs, early_stopping, tolerance, device):
    model.to(device)

    loss_train, loss_val = [], []
    acc_train, acc_val = [], []
    history1 = hl.History()
    canvas1 = hl.Canvas()

    best_val_loss = float('inf')
    best_model = None
    num_epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        tot_acc_train, tot_count_train, n_train_batches, tot_loss_train = 0, 0, 0, 0
        for _, label, _, concept in train_loader:
            concept = concept.to(device)
            label = torch.argmax(label, dim=1).to(device)
            logits = model(concept)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            tot_loss_train += loss
            
            loss.backward()
            optimizer.step()

            pred_label = torch.argmax(logits, dim=1)
            accuracy = (pred_label == label).sum().item()

            tot_acc_train += accuracy
            tot_count_train += label.size(0)
            n_train_batches += 1 

        avg_loss_train = tot_loss_train / n_train_batches
        loss_train.append(avg_loss_train)
        accuracy_train = (tot_acc_train / tot_count_train) * 100
        acc_train.append(accuracy_train)

        tot_acc_val, tot_count_val, n_val_batches, tot_loss_val = 0, 0, 0, 0

        with torch.no_grad():
            model.eval()

            for _, label, _, concept in val_loader:
                concept = concept.to(device)
                label = torch.argmax(label, dim=1).to(device)
                logits = model(concept)
                loss = criterion(logits, label)

                tot_loss_val += loss

                pred_label = torch.argmax(logits, dim=1)
                accuracy = (pred_label == label).sum().item()

                tot_acc_val += accuracy
                tot_count_val += label.size(0)
                n_val_batches += 1

            avg_loss_val = tot_loss_val / n_val_batches
            loss_val.append(avg_loss_val)
            accuracy_val = (tot_acc_val / tot_count_val) * 100
            acc_val.append(accuracy_val)

            if epoch % 1 == 0:
                history1.log(epoch, train_loss = avg_loss_train, train_accuracy = accuracy_train, val_loss = avg_loss_val, val_accuracy = accuracy_val)
            with canvas1:
                canvas1.draw_plot([history1["train_loss"], history1["val_loss"]], labels=['Training Loss', 'Validation Loss'])
                canvas1.draw_plot([history1["train_accuracy"], history1["val_accuracy"]], labels=['Training Accuracy', 'Validation Accuracy'])

            if avg_loss_val < best_val_loss and (avg_loss_train - avg_loss_val) > tolerance:
                best_val_loss = avg_loss_val
                best_model = copy.deepcopy(model)
                num_epochs_no_improve = 0
            else:
                num_epochs_no_improve += 1

            if num_epochs_no_improve >= early_stopping:
                print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4f}")
                break
                

    return loss_train, acc_train, loss_val, acc_val, best_model

MLP_model = MLP(input_dim=num_concepts, num_classes=len(ordered_labels), dropout_val=0.3, expand_dim=[256, 512, 256, 128]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MLP_model.parameters(), lr=1e-4, weight_decay=1e-5, momentum=0.7, nesterov=True)

start = timer()

torch.mps.empty_cache()

loss_train, acc_train, loss_val, acc_val, best_model = trainMLP(MLP_model, train_loader, val_loader, criterion, optimizer, num_epochs=10, early_stopping=5, tolerance=0.1, device=device)

end = timer()
print(type(acc_train), acc_train)
print(type(acc_val), acc_val)
print(type(loss_train), loss_train)
print(type(loss_val), loss_val)

plot_learning_acc_loss(loss_train, acc_train, loss_val, acc_val, 'MLPprova1')