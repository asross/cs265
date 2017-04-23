import numpy as np
import scipy.stats

class cacheprop(object):
  def __init__(self, getter): self.getter = getter
  def __get__(self, actual_self, _):
    value = self.getter(actual_self)
    actual_self.__dict__[self.getter.__name__] = value
    return value

class Workload():
  @cacheprop
  def n_unique_keys(self):
    creates = set([key for key, is_write in self.queries if is_write])
    return len(creates)

class RoundRobinWorkload(Workload):
  def __init__(self, n_queries=25000, k_classes=2500):
    self.n = n_queries
    self.k = k_classes

  @cacheprop
  def queries(self):
    return readwritify([q % self.k for q in range(self.n)])

  def __repr__(self):
    return 'RoundRobinWorkload'

class ZipfWorkload(Workload):
  def __init__(self, n_queries=25000, zipf_param=1.1):
    assert(zipf_param > 1)
    self.n = n_queries
    self.z = zipf_param

  @cacheprop
  def queries(self):
    return readwritify(np.random.zipf(self.z, self.n))

  def __repr__(self):
    return 'ZipfWorkload({})'.format(self.z)

class MultinomialWorkload(Workload):
  def __init__(self, n_queries=25000, k_classes=2500, dist=scipy.stats.uniform()):
    self.n = n_queries
    self.k = k_classes
    self.dist = dist

  @cacheprop
  def queries(self):
    self.probs = self.dist.rvs(size=self.k)
    self.probs /= self.probs.sum()
    return readwritify(np.random.choice(self.k, size=self.n, p=self.probs))

  def __repr__(self):
    return 'MultinomialWorkload({})'.format(distr(self.dist))

class UniformWorkload(MultinomialWorkload):
  def __init__(self, n_queries=25000, k_classes=2500):
    self.n = n_queries
    self.k = k_classes
    self.dist = scipy.stats.uniform()

  def __repr__(self):
    return 'UniformWorkload'

class DiscoverDecayWorkload(Workload):
  def __init__(self, n_queries=25000,
      lookups=scipy.stats.poisson(8),
      creates=scipy.stats.poisson(4),
      updates=scipy.stats.poisson(2),
      popularity=scipy.stats.beta(2,2),
      decay_rate=scipy.stats.beta(100,1)):
    self.n = n_queries
    self.lookups = lookups
    self.creates = creates
    self.updates = updates
    self.popularity = popularity
    self.decay_rate = decay_rate

  def sample(self, pops, size):
    return np.random.choice(len(pops), p=pops/pops.sum(), size=size)

  def __repr__(self):
    return 'DiscoverDecay(\nlookups~{},\ncreates~{},\nupdates~{},\npopularity~{},\ndecay_rate~{})'.format(
        *[distr(d) for d in [self.lookups,self.creates,self.updates,self.popularity,self.decay_rate]])

  @cacheprop
  def queries(self):
    queries = []
    populs = np.zeros(self.n)
    decays = np.zeros(self.n)
    k = 0

    while len(queries) < self.n:
      # newly created keys
      creates = self.creates.rvs()
      populs[k:k+creates] = self.popularity.rvs(size=creates)
      decays[k:k+creates] = self.decay_rate.rvs(size=creates)
      for i in range(creates):
        queries.append([k+i, 1])
      k += creates

      # reads/updates
      if k > 0:
        lookups = self.lookups.rvs()
        updates = self.updates.rvs()
        keys = self.sample(populs[:k], lookups + updates)
        for key, a in zip(keys, [0]*lookups + [1]*updates):
          queries.append([key, a])

      # update popularity
      populs *= decays

    return np.array(queries)

class PeriodicDecayWorkload(Workload):
  def __init__(self, n_queries=25000,
      lookups=scipy.stats.poisson(8),
      creates=scipy.stats.poisson(4),
      updates=scipy.stats.poisson(2),
      popularity=scipy.stats.beta(2,2),
      decay_rate=scipy.stats.beta(100,1),
      period=2400,
      cuspiness=2):
    self.n = n_queries
    self.lookups = lookups
    self.creates = creates
    self.updates = updates
    self.popularity = popularity
    self.decay_rate = decay_rate
    self.period = period
    self.cuspiness = cuspiness

  def sample(self, pops, size):
    return np.random.choice(len(pops), p=pops/pops.sum(), size=size)

  def __repr__(self):
    return 'PeriodicDecay(\nlookups~{},\ncreates~{},\nupdates~{},\npopularity~{},\ndecay_rate~{}\nperiod,cuspity={},{})'.format(
        *([distr(d) for d in [self.lookups,self.creates,self.updates,self.popularity,self.decay_rate]]+[self.period,self.cuspiness]))

  @cacheprop
  def queries(self):
    queries = []
    starts = np.zeros(self.n)
    populs = np.zeros(self.n)
    decays = np.zeros(self.n)
    t = 0
    k = 0

    while len(queries) < self.n:
      # newly created keys
      creates = self.creates.rvs()
      populs[k:k+creates] = self.popularity.rvs(size=creates)
      decays[k:k+creates] = self.decay_rate.rvs(size=creates)
      starts[k:k+creates] = t
      for i in range(creates):
        queries.append([k+i, 1])
      k += creates

      # reads/updates
      if k > 0:
        lookups = self.lookups.rvs()
        updates = self.updates.rvs()
        age = t - starts[:k]
        pop = populs[:k] \
            * decays[:k] ** age \
            * (1-cycloid((age % self.period)/self.period)) ** self.cuspiness
        keys = self.sample(pop, lookups + updates)
        for key, a in zip(keys, [0]*lookups + [1]*updates):
          queries.append([key, a])

      t += 1

    return np.array(queries)

def distr(dist):
  return '{}{}'.format(dist.dist.name, dist.args)

def readwritify(keys, read_fraction=1):
  seen = set()
  queries = []
  for key in keys:
    if key not in seen:
      seen.add(key)
      queries.append([key, 1])
    else:
      a = 0 if np.random.rand() < read_fraction else 1
      queries.append([key, a])
  return np.array(queries)

cycloid_t = np.linspace(0, 2*np.pi, 1000)
cycloid_x = cycloid_t - np.sin(cycloid_t)
cycloid_y = 1 - np.cos(cycloid_t)

def _cycloid(x):
  return cycloid_y[np.argmin(np.abs(cycloid_x - np.array([x*2*np.pi]).T), axis=1)] / 2.0

cyc_mapping = _cycloid(np.arange(1000)/1000.)

def cycloid(x):
  return cyc_mapping[np.floor(x*1000).astype(int)]
