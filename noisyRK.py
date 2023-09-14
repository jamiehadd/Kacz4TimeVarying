import numpy as np

#Simulates the Randomized Kaczmarz algorithm on a noisy, uncorrupted system
def RK_ex_n(A,x,b,mu,sig,num_iter):
  m = A.shape[0]
  n = A.shape[1]
  x_k = np.zeros((n,1))
  prob = [0]*m
  error = [0]*num_iter
  for j in range(m):
    prob[j] = (np.linalg.norm(A[j])**2)/(np.linalg.norm(A)**2)
  for k in range(num_iter):
    noise = 1*np.random.normal(mu, sig, size=(m,1))
    r = np.random.choice(range(m),p=prob)
    x_k_new = x_k + ((b[r][0] + noise[r] - np.inner(x_k.T,A[r,:])[0])*A[r,:].T).reshape(n,1)
    x_k = x_k_new
    error[k] = np.linalg.norm(x_k-x)**2
  return [x_k,list(range(num_iter)),error]