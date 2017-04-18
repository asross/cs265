import numpy as np

def distr(dist):
  return '{}{}'.format(dist.dist.name, dist.args)

class cacheprop(object):
  def __init__(self, getter): self.getter = getter
  def __get__(self, actual_self, _):
    value = self.getter(actual_self)
    actual_self.__dict__[self.getter.__name__] = value
    return value

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
