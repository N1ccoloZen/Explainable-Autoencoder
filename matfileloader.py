import scipy.io
import sys
import numpy as np
import pandas as pd

name = sys.argv[1]

mat = scipy.io.loadmat(name)

print(mat.keys())

anno = mat['anno']
#df = pd.DataFrame(matrix)
#np.set_printoptions(threshold=np.inf, linewidth=np.inf)

#df = pd.DataFrame(mat)

#print(anno.shape)

anno_struct = anno[0, 0]

print(anno_struct.dtype.names)

print(anno_struct['imname'])

print(anno_struct['objects'])
