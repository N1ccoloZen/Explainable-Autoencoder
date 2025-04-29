import scipy.io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name = sys.argv[1]

mat = scipy.io.loadmat(name)
print("Key features:", mat.keys())
#print(mat.keys())
anno = mat['anno']
#df = pd.DataFrame(matrix)
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)
#df = pd.DataFrame(mat)
#print(anno.shape)

anno_struct = anno[0, 0]

print(anno_struct.dtype.names)

print("Image name:", anno_struct['imname'])

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
            print("This mask exists")
            print("Concept ", label, "exists in photo:", anno_struct['imname'])

        #if label == 'head':
            #print("This is the mask:")
            #mask = part[1]
            #if mask.sum() > 0:
            #   print("This mask exists")
            #print(f"Shape: {mask.shape}")
            #plt.imshow(mask, cmap='gray')
            #plt.show()

print("detected obj:", anno_struct['objects'].size)




