import numpy as np

class ZipfWorkload():
  def __init__(self, n, z):
    self.n = n
    self.z = z

  def generate(self):
    return np.random.zipf(self.z, self.n)

# dependent dirichlet processes
class ChineseRestaurantProcessWorkload():

class RottingChineseRestaurantProcessWorkload():
  def __init__(self, n, decay_rate, popularity_prior=None):
    self.n = n
    if popularity_prior is None:
      popularity_prior = np.random.random # uniform
    self.decay_rate = decay_rate
    self.popu_prior = popularity_prior

  def generate(self):
    queries = np.zeros(self.n)
    for t in range(self.n):

    return queries

