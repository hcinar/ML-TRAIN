"""
What versions we need to complete series
"""

#Imports Start
import sys
import numpy as np
import matplotlib
import nnfs
#Imports End

#Version Info Start
print("Python: ", sys.version) #must > 3.7.7
print("Numpy:", np.__version__) #must > 1.18.2
print("Matplotlib:", matplotlib.__version__) #must >3.2.1
print("NNFS:", nnfs.__version__)  #could be > 0.5.1  
#Version Info End

