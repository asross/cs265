import numpy as np
import scipy.stats

def distr(dist):
  return '{}{}'.format(dist.dist.name, dist.args)

class cacheprop(object):
  def __init__(self, getter): self.getter = getter
  def __get__(self, actual_self, _):
    value = self.getter(actual_self)
    actual_self.__dict__[self.getter.__name__] = value
    return value

class RoundRobinWorkload():
  def __init__(self, n_queries=25000, k_classes=2500):
    self.n = n_queries
    self.k = k_classes

  @cacheprop
  def queries(self):
    return np.array([q % self.k for q in range(self.n)])

  def __str__(self):
    return 'RoundRobinWorkload'

class ZipfWorkload():
  def __init__(self, n_queries=25000, zipf_param=1.1):
    assert(zipf_param > 1)
    self.n = n_queries
    self.z = zipf_param

  @cacheprop
  def queries(self):
    return np.random.zipf(self.z, self.n)

  def __str__(self):
    return 'ZipfWorkload({})'.format(self.z)

class MultinomialWorkload():
  def __init__(self, n_queries=25000, k_classes=2500, dist=scipy.stats.uniform()):
    self.n = n_queries
    self.k = k_classes
    self.dist = dist

  @cacheprop
  def queries(self):
    self.probs = self.dist.rvs(size=self.k)
    self.probs /= self.probs.sum()
    return np.random.choice(self.k, size=self.n, p=self.probs)

  def __str__(self):
    return 'MultinomialWorkload({})'.format(distr(self.dist))

class UniformWorkload(MultinomialWorkload):
  def __init__(self, n_queries=25000, k_classes=2500):
    self.n = n_queries
    self.k = k_classes
    self.dist = scipy.stats.uniform()

  def __str__(self):
    return 'UniformWorkload'

class DiscoverDecayWorkload():
  def __init__(self, n_queries=25000,
      discoveries=scipy.stats.poisson(3),
      popularity=scipy.stats.beta(2,2),
      decay_rate=scipy.stats.beta(100,1)):
    self.n = n_queries
    self.discoveries = discoveries
    self.popularity = popularity
    self.decay_rate = decay_rate

  def sample(self, pops):
    p = np.array(pops)
    p /= p.sum()
    return np.random.choice(len(p), p=p, size=10)

  def __str__(self):
    return 'DiscoverDecay(\nnew keys~{},\npopularity~{},\ndecayrate~{})'.format(
        *[distr(d) for d in [self.discoveries,self.popularity,self.decay_rate]])

  @cacheprop
  def queries(self):
    queries = []
    populs = []
    decays = []
    while len(queries) < self.n:
      for i in range(self.discoveries.rvs()):
        populs.append(self.popularity.rvs())
        decays.append(self.decay_rate.rvs())
      if len(populs) > 0:
        for el in self.sample(populs):
          queries.append(el)
        populs = [populs[i] * decays[i] for i in range(len(populs))]
    return np.array(queries)
