import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random


from scipy.linalg import svdvals
from sklearn.preprocessing import normalize
from minsv import *

def error_bound_thm_2_4(A,x,q,beta,N,NBS,p,num_iter):
  m = A.shape[0]
  n = A.shape[1]

  x_1 = np.zeros((n,1))
  x_1[0] = 1
  #sigmin = minSV1(A,x_1,q,beta)
  sigmin = (q-beta)*np.sqrt((q-beta)*m/n)
  sigmax = np.linalg.svd(A,compute_uv=False).max()

  term_1 = ((sigmin**2)/(q*m))*((q-beta)/(q))
  term_2 = ((sigmax**2)/(q*m))*( (2*np.sqrt(beta*(1-beta)))/(1-q-beta)  +  (beta*(1-beta))/((1-q-beta)**2) )
  term_3 = ((sigmax)/(q*m))*( (2*np.sqrt(beta*m))/(m*(1-q-beta))  +  (2*beta*np.sqrt(m*(1-beta)))/(m*((1-q-beta)**2)) )
  term_4 = (beta)/(q*(m**2)*((1-q-beta)**2))

  phi = term_1 - term_2 - term_3
  c_2 = term_3 + term_4
  e_0 = np.linalg.norm(-1*x)**2

  error = [0]*num_iter
  error[0] = e_0**2
  for i in range(1,num_iter):
    t2 = (np.linalg.norm(NBS[i],2)**2)/((NBS[i]).shape[0])
    #modify
    t3 = c_2*(np.linalg.norm(N[i],1)**2)
    error[i] = (1-p*phi)*error[i-1] + p*(t2+t3)
  return [error,phi]