import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from gencorrup import *
from gennoise import *

from scipy.linalg import svdvals
from sklearn.preprocessing import normalize

#Algorithm 2 of QRK, carried out on a system under the influence of mean 0 noise,
#as well as corruption of a given rate and size
def QuantileRK2(A,x,b,q,t,beta,corr_size,sig,num_iter):
  m = A.shape[0]
  n = A.shape[1]
  x_k = np.zeros((n,1))
  error = [0]*num_iter
  max_noise = 0
  N = []
  NBS = []
    
  for k in range(num_iter):
    # this will be the corrupted b that we use
    C = generateCorruption_s(b,beta,corr_size)
    b_k = C[0]
    
    #b_k = b_k + np.random.normal(0,sig,(m,1))
    b_k,noise,noisemax_k = generateNoise(b_k,0,sig)
    
    N.append(noise)
    
    if noisemax_k > max_noise:
        max_noise = noisemax_k

    size_noise = np.linalg.norm(noise,1)
    # now we calculate quantile from sampled rows
    residuals = np.abs(np.dot(A,x_k) - b_k)

    # calculate the q-quantile of the set of residuals
    q_quantile = np.quantile(residuals,q)

    # get a list containing the indices of the admissible rows
    B = np.where(residuals <= q_quantile)[0]
    
    # determine the set of uncorrupted admissible rows, and select the noise from those elements
    B_S = B[B not in C[1]]
    noise_B_S = noise[B_S].reshape(noise[B_S].shape[1],1)
    NBS.append(noise_B_S)
    
    # get a row where current residual < q_quantile
    i = np.random.choice(B)

    x_k_new = x_k - ((np.inner(x_k.T,A[i,:])[0] - b_k[i][0])*A[i,:].T).reshape(n,1)
    x_k = x_k_new
    error[k] = np.linalg.norm(x_k - x)**2
  return [x_k,list(range(num_iter)),error,max_noise,N,NBS]