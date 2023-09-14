import numpy as np

#Bounds the error of a noisy RK run using the function from Theorem 2.5
def error_bound_thm_2_5(A,mu,sig,num_iter,sigmin,e_0):
  error = [0]*num_iter

  phi2 = 1 - (sigmin/np.linalg.norm(A))**2
  error[0] = e_0**2

  for k in range(1,num_iter):
    error[k] = (((1-phi2**(k+1))/(1-phi2))*(mu**2+sig**2))*(A.shape[0]/np.linalg.norm(A)**2) + (phi2**(k+1))*e_0**2
  return error