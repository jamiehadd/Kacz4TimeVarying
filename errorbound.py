import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from minsv import *

from scipy.linalg import svdvals
from sklearn.preprocessing import normalize

#Plots the error bound from Corollary 1, applied to a noisy corrupted system that runs QRK
def error_bound_crl_1(A,x,q,beta,n_max,p,num_iter):
  m = A.shape[0]
  n = A.shape[1]

  x_1 = np.zeros((n,1))
  x_1[0] = 1
  #sigmin = minSV2(A,q,beta)
  sigmin = (q-beta)*np.sqrt((q-beta)*m/n)
  sigmax = np.linalg.svd(A,compute_uv=False).max()

  term_1 = ((sigmin**2)/(q*m))*((q-beta)/(q))
  term_2 = ((sigmax**2)/(q*m))*(((2*np.sqrt(beta*(1-beta)))/((1-q-beta)))  +  ((beta*(1-beta))/((1-q-beta)**2)))
  term_3 = ((sigmax)/(q*m))*(((2*np.sqrt(beta*m))/(m*(1-q-beta)))  +  ((2*beta*np.sqrt(m*(1-beta)))/(m*((1-q-beta)**2))))
  term_4 = (beta)/(q*(m**2)*((1-q-beta)**2))

  phi = term_1 - term_2 - term_3
  c_2 = term_3 + term_4
  e_0 = np.linalg.norm(-1*x)**2

  error = [0]*num_iter
  error[0] = e_0**2
  for i in range(1,num_iter):
    c_a = (1-p*phi)**(i+1)
    error[i] = c_a*e_0**2 + ((1-c_a)/phi)*(n_max**2)*(1+(c_2*(m**2)))
  return error
     
#Plots the error bound from Corollary 2, applied to a noisy corrupted system that runs QRK
def error_bound_crl_2(A,x,q,beta,mu,sig,mu2,sig2,p,num_iter):
  m = A.shape[0]
  n = A.shape[1]

  x_1 = np.zeros((n,1))
  x_1[0] = 1
  sigmin = (q-beta)*np.sqrt((q-beta)*m/n)
  sigmax = np.linalg.svd(A,compute_uv=False).max()

  term_1 = ((sigmin**2)/(q*m))*((q-beta)/(q))
  term_2 = ((sigmax**2)/(q*m))*(((2*np.sqrt(beta*(1-beta)))/((1-q-beta)))  +  ((beta*(1-beta))/((1-q-beta)**2)))
  term_3 = ((sigmax)/(q*m))*(((2*np.sqrt(beta*m))/(m*(1-q-beta)))  +  ((2*beta*np.sqrt(m*(1-beta)))/(m*((1-q-beta)**2))))
  term_4 = (beta)/(q*(m**2)*((1-q-beta)**2))

  phi = term_1 - term_2 - term_3
  c_2 = term_3 + term_4
  e_0 = np.linalg.norm(-1*x)**2

  error = [0]*num_iter
  error[0] = e_0**2
  for i in range(1,num_iter):
    c_a = (1-p*phi)**(i+1)
    error[i] = c_a*e_0**2 + ((1-c_a)/phi)*((mu**2) + (sig**2) + c_2*((m**2)*(mu2**2) + m*(sig2**2)))
  return error
     
#Plots the error bound from Corollary 3, applied to a noisy corrupted system that runs QRK
def error_bound_crl_3(A,x,q,beta,sig,p,num_iter):
  m = A.shape[0]
  n = A.shape[1]

  x_1 = np.zeros((n,1))
  x_1[0] = 1
  sigmin = (q-beta)*np.sqrt((q-beta)*m/n)
  sigmax = np.linalg.svd(A,compute_uv=False).max()

  term_1 = ((sigmin**2)/(q*m))*((q-beta)/(q))
  term_2 = ((sigmax**2)/(q*m))*(((2*np.sqrt(beta*(1-beta)))/((1-q-beta)))  +  ((beta*(1-beta))/((1-q-beta)**2)))
  term_3 = ((sigmax)/(q*m))*(((2*np.sqrt(beta*m))/(m*(1-q-beta)))  +  ((2*beta*np.sqrt(m*(1-beta)))/(m*((1-q-beta)**2))))
  term_4 = (beta)/(q*(m**2)*((1-q-beta)**2))

  phi = term_1 - term_2 - term_3
  c_2 = term_3 + term_4
  e_0 = np.linalg.norm(-1*x)**2

  error = [0]*num_iter
  error[0] = e_0**2
  for i in range(1,num_iter):
    c_a = (1-p*phi)**(i+1)
    error[i] = c_a*e_0**2 + ((1-c_a)/phi)*(sig**2)*(1+(c_2*((m**2)*(2/np.pi)  + m*(1 - (2/np.pi)))))
  return error