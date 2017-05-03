import numpy as np
from cache import *
from layer import *
from bloom_assignments import *

default_layer_ratio = 2
default_memtbl_size = 100

class LSMulator():
  def __init__(self, cache_size=50, memtbl_size=default_memtbl_size, layer_ratio=default_layer_ratio, bloom_size=100, page_size=256):
    self.cache = Cache(cache_size)
    self.memtbl = Layer(memtbl_size, ratio=layer_ratio, bsize=bloom_size, index=0)
    self.page_size = 256

  def put(self, key):
    self.memtbl.put(key)

  def get(self, key):
    if key not in self.memtbl.entries and self.cache.get(key):
      return True
    else:
      return self.memtbl.get(key)

  def reset_counters(self):
    for component in [self.cache, self.memtbl] + self.layers:
      component.reset_counters()

  @property
  def layer_sizes(self):
    return np.array([l.size for l in self.layers])

  @property
  def layers(self):
    return self.memtbl.children()

  @property
  def disk_accesses(self):
    return sum(l.disk_accesses(self.page_size) for l in self.layers)

  @classmethod
  def emulate(kls, queries, **kwargs):
    lsmtree = kls(**kwargs)
    for key, is_write in queries:
      if is_write:
        lsmtree.put(key)
      else:
        lsmtree.get(key)
    return lsmtree

  @classmethod
  def cache_vs_bloom(kls, workload, total, dM=100, balloc=monkey_assignment, layer_ratio=2, memtbl=default_memtbl_size):
    trees = []
    layers = LSMulator.emulate(workload.queries, memtbl_size=memtbl, layer_ratio=layer_ratio).layer_sizes
    for bloom in range(0, total, dM):
      trees.append(LSMulator.emulate(workload.queries,
        layer_ratio=layer_ratio,
        memtbl_size=memtbl,
        cache_size=total - bloom,
        bloom_size=balloc(bloom, memtbl, layers)))
    return trees

  @classmethod
  def cache_vs_bloom_vs_buf(kls, workload, total, dM=100, balloc=monkey_assignment, layer_ratio=2, verbose=False):
    trees = []
    for memtbl in range(dM, total, dM):
      if verbose:
        print('Memtable =', memtbl)
      layers = LSMulator.emulate(workload.queries, memtbl_size=memtbl, layer_ratio=layer_ratio).layer_sizes
      for bloom in range(0, total - memtbl, dM):
        trees.append(LSMulator.emulate(workload.queries,
          layer_ratio=layer_ratio,
          memtbl_size=memtbl,
          cache_size=total - memtbl - bloom,
          bloom_size=balloc(bloom, memtbl, layers)))
    return trees

if __name__ == '__main__':
  import pdb
  from workloads import readwritify
  queries = readwritify(np.random.zipf(1.5, 100000), read_fraction=0.95)
  lsmtree = LSMulator.emulate(queries)
  pdb.set_trace()
  pass
