import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random

from scipy.linalg import svdvals
from sklearn.preprocessing import normalize

#Generates noise of a given mean and standard deviation for a linear system Ax=b.
#Returns the result of b shifted by noise, as well as the maximum element in the noise vector
def generateNoise(b,mu,sig):
  m_b = b.shape[0]
  noise = 1*np.random.normal(mu, sig, size=(m_b,1))
  noise_max = np.absolute(noise).max()
  return [b + noise, noise, noise_max]