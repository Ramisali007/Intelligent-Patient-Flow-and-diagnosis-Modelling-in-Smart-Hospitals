import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

try:
    import streamlit
    print("Streamlit version:", streamlit.__version__)
except ImportError as e:
    print("Error importing streamlit:", e)

try:
    import pandas
    print("Pandas version:", pandas.__version__)
except ImportError as e:
    print("Error importing pandas:", e)

try:
    import numpy
    print("NumPy version:", numpy.__version__)
except ImportError as e:
    print("Error importing numpy:", e)

try:
    import matplotlib
    print("Matplotlib version:", matplotlib.__version__)
except ImportError as e:
    print("Error importing matplotlib:", e)

try:
    import plotly
    print("Plotly version:", plotly.__version__)
except ImportError as e:
    print("Error importing plotly:", e)

try:
    import scipy
    print("SciPy version:", scipy.__version__)
except ImportError as e:
    print("Error importing scipy:", e)

try:
    import sklearn
    print("Scikit-learn version:", sklearn.__version__)
except ImportError as e:
    print("Error importing scikit-learn:", e)

try:
    import statsmodels
    print("Statsmodels version:", statsmodels.__version__)
except ImportError as e:
    print("Error importing statsmodels:", e)

print("All imports tested.")
