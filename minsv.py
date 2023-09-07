import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random

from scipy.linalg import svdvals
from sklearn.preprocessing import normalize

def minSV1(A,x,q,beta):
    m_in = A.shape[0]
    inner_p = [0]*m_in
    for i in range(m_in):
        inner_p[i] = [np.inner(x.T,A[i]).item(0)**2,i]
    df_inner = pd.DataFrame(inner_p)
    df_asc = df_inner.sort_values(by=[0])
    df_min = df_asc.iloc[:int(m_in*(q-beta))]
    min_arr = np.array(df_min[1])
    A_df = pd.DataFrame(A).loc[min_arr]
    A_S = np.asmatrix(A_df)
    return (np.linalg.norm(A_S*x))/(np.linalg.norm(x))

def minSV2(A,q,beta):
    min_sv = A.shape[0] #initialize to something that will be greater than any singular value
    for i in range(100):
      A_S = A[np.random.choice(A.shape[0], size=np.floor((((1-q)-beta)*A.shape[0])).astype(int), replace=False),]
      min_sv = min(min_sv,min(svdvals(A_S)))
    return min_sv