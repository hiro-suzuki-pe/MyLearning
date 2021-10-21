import numpy as np
from scipy import sparse


x = np.array([[1,2,3],[4,5,6]])
print('x:\n{}'.format(x))

eye = np.eye(4)
print('numpy arry:\n{}'.format(eye))

sparse_matrix = sparse.csr_matrix(eye)
print('\nSciPy sparse CSR matrix:\n{}'.format(sparse_matrix))

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data,(row_indices, col_indices)))
print('COO representation:\n{}'.format(eye_coo))

#%matplotlib inline
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)
y = np.sin(x)
#plt.plot(x,y,marker='x')
#plt.show()

import pandas as pd
data = {'Name': ['John', 'Anbna', 'Peter', 'Linda'],
        'Location': ['New York', 'Paris', 'Berlin', 'London'],
        'Age':[24,13,53,33]
        }

from IPython.display import display
data_pandas = pd.DataFrame(data)

#display(data_pandas)

import sys
import matplotlib
import scipy as sp
import IPython
import sklearn

print("Python version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("matplotlib version: {}".format(matplotlib.__version__))
print("NumPy version: {}".format(np.__version__))
print("scipy version: {}".format(sp.__version__))
print("IPython version: {}".format(IPython.__version__))
print("sklearn version: {}".format(sklearn.__version__))



