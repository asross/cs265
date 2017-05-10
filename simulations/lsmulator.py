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
    self.layer_queries = 0
    self.puts = 0
    self.gets = 0

  def put(self, key):
    self.puts += 1
    self.memtbl.put(key)

  def get(self, key):
    if key in self.memtbl.entries:
      self.memtbl.hits += 1
      return True
    elif self.cache.get(key):
      return True
    else:
      self.layer_queries += 1
      result = self.memtbl.get(key)
      if result:
        self.cache.put(key)
      return result

  def reset_counters(self):
    for component in [self.cache, self.memtbl] + self.layers:
      component.reset_counters()

  def bigger_cache_savings(self, dM=1):
    return dM * self.cache.last_slot_hits * (self.disk_accesses / self.layer_queries)

  def bigger_memtbl_savings(self, dM=1):
    T = self.memtbl.ratio
    M = self.memtbl.size
    return (-np.log(M / (M + dM)) / np.log(T)) * (self.puts-self.dupes_squashed)

  def bigger_bloom_savings(self, dM=1, ballocs=monkey_assignment, bits_per_key=64):
    bc = np.array([l.bloom.bit_length for l in self.layers])

    total_bloom_bits = bc.sum()
    total_bloom_mem = int(round(total_bloom_bits / bits_per_key))

    b1 = ballocs(total_bloom_mem,    self.memtbl.size, self.layer_sizes, bits_per_key=bits_per_key)
    b2 = ballocs(total_bloom_mem+dM, self.memtbl.size, self.layer_sizes, bits_per_key=bits_per_key)

    if not np.allclose(b1, bc):
      print('Warning: current bloom allocation is different than', ballocs.__name__)

    das1 = sum([l.bloom.est_disk_accesses(m1) for m1, l in zip(b1, self.layers)])
    das2 = sum([l.bloom.est_disk_accesses(m2) for m2, l in zip(b2, self.layers)])

    return das1 - das2

  @property
  def layer_sizes(self):
    return np.array([l.size for l in self.layers])

  @property
  def layers(self):
    return self.memtbl.children()

  @property
  def disk_accesses(self):
    return sum(l.disk_accesses(self.page_size) for l in self.layers)

  @property
  def dupes_squashed(self):
    return sum(l.dupes_squashed for l in self.layers)

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
  def cache_vs_bloom(kls, workload, total, dM=100, ballocs=monkey_assignment, layer_ratio=2, memtbl=default_memtbl_size):
    trees = []
    layers = LSMulator.emulate(workload.queries, memtbl_size=memtbl, layer_ratio=layer_ratio).layer_sizes
    for bloom in range(0, total, dM):
      trees.append(LSMulator.emulate(workload.queries,
        layer_ratio=layer_ratio,
        memtbl_size=memtbl,
        cache_size=total - bloom,
        bloom_size=ballocs(bloom, memtbl, layers)))
    return trees

  @classmethod
  def cache_vs_bloom_vs_buf(kls, workload, total, dM=100, ballocs=monkey_assignment, layer_ratio=2, verbose=False):
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
          bloom_size=ballocs(bloom, memtbl, layers)))
    return trees

if __name__ == '__main__':
  import pdb
  from workloads import readwritify
  queries = readwritify(np.random.zipf(1.5, 100000), update_fraction=0.05, null_read_fraction=0.01)
  lsmtree = LSMulator.emulate(queries, bloom_size=4096)
  pdb.set_trace()
  pass
