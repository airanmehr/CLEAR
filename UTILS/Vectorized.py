import numpy as np
import pandas as pd

import numba


@numba.vectorize
def vectorizedLog(x):
    return float(np.log(x))
def numbaLog(df):
    return  pd.DataFrame(vectorizedLog(df.values),columns=df.index,index=df.index).astype(float)

@numba.vectorize
def vectorizedExp(x):
    return float(np.exp(x))
def numbaExp(df):
    return  pd.DataFrame(vectorizedExp(df.values),columns=df.index,index=df.index).astype(float)