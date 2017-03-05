import numpy as np
import scipy.stats

class RoundRobinWorkload():
  def __init__(self, n_queries=25000, k_classes=2500):
    self.n = n_queries
    self.k = k_classes

  def generate(self):
    return np.array([q % self.k for q in range(self.n)])

class ZipfWorkload():
  def __init__(self, n_queries=25000, zipf_param=1.1):
    assert(zipf_param > 1)
    self.n = n_queries
    self.z = zipf_param

  def generate(self):
    return np.random.zipf(self.z, self.n)

class MultinomialWorkload():
  def __init__(self, n_queries=25000, k_classes=2500, dist=scipy.stats.uniform()):
    self.n = n_queries
    self.k = k_classes
    self.dist = dist

  def generate(self):
    probs = self.dist.rvs(size=self.k)
    probs /= probs.sum()
    return np.random.choice(self.k, size=self.n, p=probs)

class TimeDecayBetaMultWorkload():
  def __init__(self, n_queries=25000,
      new_discoveries=scipy.stats.poisson(3),
      new_popularity=scipy.stats.beta(2,2),
      new_decay_rate=scipy.stats.beta(100,1),
      new_time_inter=scipy.stats.expon(0.25)):
    self.n = n_queries
    self.new_discoveries = new_discoveries
    self.new_popularity = new_popularity
    self.new_decay_rate = new_decay_rate
    self.new_time_inter = new_time_inter

  def sample(self, pops):
    p = np.array(pops)
    p /= p.sum()
    return np.random.choice(len(p), p=p, size=10)

  def generate(self):
    queries = []
    populs = []
    decays = []
    while len(queries) < self.n:
      for i in range(self.new_discoveries.rvs()):
        populs.append(self.new_popularity.rvs())
        decays.append(self.new_decay_rate.rvs())
      if len(populs) > 0:
        for el in self.sample(populs):
          queries.append(el)
        dt = self.new_time_inter.rvs()
        populs = [populs[i] * decays[i]**dt for i in range(len(populs))]
    return np.array(queries)
