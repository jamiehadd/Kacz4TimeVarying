import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from gencorrup import *

from scipy.linalg import svdvals
from sklearn.preprocessing import normalize

#The code of algorithm 2 of QRK
def QuantileRK2_ex(A,x,b,q,t,beta,corr_size,num_iter):
  m = A.shape[0]
  n = A.shape[1]
  x_k = np.zeros((n,1))
  error = [0]*num_iter
  for k in range(num_iter):
    # this will be the corrupted b that we use
    b_k = generateCorruption_s(b,beta,corr_size)

    # now we calculate quantile from sampled rows
    residuals = np.abs(np.dot(A,x_k) - b_k)

    # calculate the q-quantile of the set of residuals
    q_quantile = np.quantile(residuals,q)

    # get a row where current residual < q_quantile
    i = np.random.choice(np.where(residuals <= q_quantile)[0])

    x_k_new = x_k - ((np.inner(x_k.T,A[i,:])[0] - b_k[i][0])*A[i,:].T).reshape(n,1)
    x_k = x_k_new
    error[k] = np.linalg.norm(x_k - x)**2
  return [x_k,list(range(num_iter)),error]

#Algorithm 2 of QRK, carried out on a system under the influence of mean 0 noise
def QuantileRK2_ex_n(A,x,b,q,t,beta,corr_size,sig,num_iter):
  m = A.shape[0]
  n = A.shape[1]
  x_k = np.zeros((n,1))
  error = [0]*num_iter
  for k in range(num_iter):
    # this will be the corrupted b that we use
    b_k = generateCorruption_s(b,beta,corr_size)
    b_k = b_k + np.random.normal(0,sig,(m,1))

    # now we calculate quantile from sampled rows
    residuals = np.abs(np.dot(A,x_k) - b_k)

    # calculate the q-quantile of the set of residuals
    q_quantile = np.quantile(residuals,q)

    # get a row where current residual < q_quantile
    i = np.random.choice(np.where(residuals <= q_quantile)[0])

    x_k_new = x_k - ((np.inner(x_k.T,A[i,:])[0] - b_k[i][0])*A[i,:].T).reshape(n,1)
    x_k = x_k_new
    error[k] = np.linalg.norm(x_k - x)**2
  return [x_k,list(range(num_iter)),error]