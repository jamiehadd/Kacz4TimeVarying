import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random

from scipy.linalg import svdvals
from sklearn.preprocessing import normalize


def generateCorruption_s(b,corruption_rate,corruption_size):
  m_b = b.shape[0]
  nums = np.zeros(m_b)
  nums[:int(m_b*corruption_rate)] = 1
  np.random.shuffle(nums)
  corruption = np.asarray(nums).reshape(m_b,1)*(corruption_size)
  return b + corruption