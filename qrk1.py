import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from gencorrup import *

from scipy.linalg import svdvals
from sklearn.preprocessing import normalize

# this is the actual code for Algorithm 1
def QuantileRK1(A,x,b,q,t,beta,corr_size,num_iter):
  m = A.shape[0]
  n = A.shape[1]
  x_j = np.zeros((n,1))
  error = [0]*num_iter
  for j in range(num_iter):
    #moved to front
    error[j] = np.linalg.norm(x_j - x)**2
    
    # this will be the corrupted b that we use
    b_k = generateCorruption_s(b,beta,corr_size)

    # sample i_1,..., i_t ~ Uniform(1,...,m)
    sampled_rows = random.sample(range(m),t)

    # sample k ~ Uniform(1,...,m)
    k = np.random.choice(range(m))

    # now we calculate quantile from sampled rows

    residuals = [None]*t
    residuals_index = 0
    for row in sampled_rows:
      # calculate the residual for that row
      residual = np.abs(np.inner(x_j.T,A[row,:])[0] - b_k[row][0])
      residuals[residuals_index] = residual
      residuals_index += 1

    # calculate the q-quantile of the set of residuals
    q_quantile = np.quantile(residuals,q)

    # calculate the current residual

    current_residual = np.abs(np.inner(x_j.T,A[k,:])[0] - b_k[k][0])

    # finally, quantile kaczmarz
    if current_residual <= q_quantile:
      x_j_new = x_j - ((np.inner(x_j.T,A[k,:])[0] - b_k[k][0])*A[k,:].T).reshape(n,1)
      x_j = x_j_new
    else:
      pass
    
  return [x_j,list(range(num_iter)),error]


# we are always using t = m, so let's make this a little faster
def QuantileRK1_ex(A,x,b,q,t,beta,corr_size,num_iter):
  m = A.shape[0]
  n = A.shape[1]
  x_j = np.zeros((n,1))
  error = [0]*num_iter
  for j in range(num_iter):
    #moved to front
    error[j] = np.linalg.norm(x_j - x)**2
    
    # this will be the corrupted b that we use
    b_k = generateCorruption_s(b,beta,corr_size)

    # sample k ~ Uniform(1,...,m)
    k = np.random.choice(range(m))

    # now we calculate quantile from sampled rows

    residuals = np.abs(np.dot(A,x_j) - b_k)

    # calculate the q-quantile of the set of residuals
    q_quantile = np.quantile(residuals,q)

    # calculate the current residual

    current_residual = np.abs(np.inner(x_j.T,A[k,:])[0] - b_k[k][0])

    # finally, quantile kaczmarz
    if current_residual <= q_quantile:
      x_j_new = x_j - ((np.inner(x_j.T,A[k,:])[0] - b_k[k][0])*A[k,:].T).reshape(n,1)
      x_j = x_j_new
    else:
      pass
  return [x_j,list(range(num_iter)),error]