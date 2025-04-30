import scipy.io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dirName = sys.argv[1]

concepts_list = []

for i, files in enumerate(os.listdir(dirName)):
    print("File name:", files)

    if  files == '2008_000652.mat':
        filename = os.path.join(dirName, files)
        mat = scipy.io.loadmat(filename)
        print("Key features:", mat.keys())
        anno = mat['anno']
        anno_struct = anno[0, 0]

        print(anno_struct.dtype.names)

        print("Image name:", anno_struct['imname'])
        break

print(i)

""" mat = scipy.io.loadmat(dirName)
print("Key features:", mat.keys()) """

""" anno = mat['anno']

anno_struct = anno[0, 0]

print(anno_struct.dtype.names)

print("Image name:", anno_struct['imname'])  """

for i in range(anno_struct['objects'].shape[1]):
    #print(anno_struct['objects'][0,i][3][0])
    #print("Object", i, ":", anno_struct['objects'][0,i])
    print("Object", i ,"name:", anno_struct['objects'][0,i][0][0])
    parts = anno_struct['objects'][0,i][3]
    print(parts.shape)
    for i in range(parts.shape[1]):
        part = parts[0, i]
        label = part[0][0]

        mask = part[1]
        if mask.sum() > 0:
            #print("This mask exists")
            print("Concept", label, "exists in photo:", anno_struct['imname'])

        #if label == 'head':
            #print("This is the mask:")
            #mask = part[1]
            #if mask.sum() > 0:
            #   print("This mask exists")
            #print(f"Shape: {mask.shape}")
            #plt.imshow(mask, cmap='gray')
            #plt.show()

print("detected obj:", anno_struct['objects'].size)
