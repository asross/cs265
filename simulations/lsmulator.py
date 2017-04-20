import numpy as np
import pdb
from collections import defaultdict

class LSMComponent():
  def __init__(self, size):
    self.hits = 0
    self.misses = 0
    self.size = size
    self.entries = []

  @property
  def full(self):
    return len(self.entries) == self.size

  @property
  def empty(self):
    return len(self.entries) == 0

  @property
  def free_space(self):
    return self.size - len(self.entries)

  @property
  def accesses(self):
    return self.hits + self.misses

  @property
  def hit_frequency(self):
    return self.hits / self.accesses

  @property
  def miss_frequency(self):
    return self.misses / self.accesses

class Cache(LSMComponent):
  def get(self, key):
    if self.size == 0:
      return False

    if key in self.entries:
      self.hits += 1
      result = True
      self.entries.remove(key)
    else:
      self.misses += 1
      result = False
      if self.full:
        self.entries.remove(self.entries[-1])
    self.entries.insert(0, key)
    return result

class Layer(LSMComponent):
  def __init__(self, size, ratio=2, index=0, bsize=100):
    super(Layer, self).__init__(size)
    self.child = None
    self.ratio = ratio
    self.index = index
    self.bsize = bsize
    self.bloom = Bloom(size, bsize, index) if index else None

  def put(self, key):
    if self.full:
      self.merge_down()
    self.entries.append(key)
    if self.bloom is not None:
      self.bloom.put(key)

  def merge(self, entries):
    assert(len(entries) <= self.size)
    if len(entries) > self.free_space:
      self.merge_down()
    self.entries += entries
    self.entries = list(set(self.entries))
    if self.bloom is not None:
      for key in entries:
        self.bloom.put(key)

  def get(self, key):
    if key in self.entries:
      self.hits += 1
      return True
    else:
      if self.bloom and self.bloom.get(key):
        self.misses += 1
      if self.child is None:
        return False
      else:
        return self.child.get(key)

  def merge_down(self):
    # if we can't accept all the new entries, then merge existing entries to the next layer
    if self.child is None: # creating it if it does not exist
      self.child = Layer(self.size*self.ratio, self.ratio, bsize=self.bsize, index=self.index+1)
    self.child.merge(self.entries)
    self.entries = []
    if self.bloom:
      self.bloom.reset()

  def children(self):
    layers = []
    layer = self.child
    while layer is not None:
      layers.append(layer)
      layer = layer.child
    return layers

class Bloom(LSMComponent):
  def __init__(self, size, bit_length=100, index=0, int_width=32):
    super(Bloom, self).__init__(size)
    if callable(bit_length):
      bit_length = bit_length(index)
    elif isinstance(bit_length, (list, tuple, np.ndarray)):
      bit_length = bit_length[index-1]
    bit_length *= int_width # to account for integer size
    self.bit_length = bit_length # m
    self.hash_count = int(np.ceil((bit_length / size) * np.log(2))) # k
    self.reset()

  def reset(self):
    self.hashes = set()
    self.hash_for = defaultdict(self.random_hash_eval)

  def random_hash_eval(self):
    return tuple(np.random.choice(self.bit_length) for _ in range(self.hash_count))

  def put(self, key):
    self.entries.append(key)
    self.hashes.update(self.hash_for[key])

  def get(self, key):
    result = self.hashes.issuperset(self.hash_for[key])
    if result and key in self.entries:
      self.misses += 1
    else:
      self.hits += 1
    return result

  @property
  def false_positive_rate(self):
    return self.miss_frequency

class LSMulator():
  def __init__(self, cache_size=50, layer_size=100, layer_ratio=2, bloom_size=100):
    self.cache = Cache(cache_size)
    self.memtbl = Layer(layer_size, ratio=layer_ratio, bsize=bloom_size, index=0)

  def put(self, key):
    self.memtbl.put(key)

  def get(self, key):
    if key not in self.memtbl.entries and self.cache.get(key):
      return True
    else:
      return self.memtbl.get(key)

  @property
  def layer_sizes(self):
    return [l.size for l in self.layers]

  @property
  def layers(self):
    return self.memtbl.children()

  @property
  def disk_accesses(self):
    return sum(l.total_accesses() for l in self.layers)

def lsmulate(queries, **kwargs):
  lsmtree = LSMulator(**kwargs)
  for key, is_write in queries:
    if is_write:
      lsmtree.put(key)
    else:
      lsmtree.get(key)
  return lsmtree

if __name__ == '__main__':
  from workloads import readwritify
  queries = readwritify(np.random.zipf(1.5, 100000), read_fraction=0.95)
  lsmtree = lsmulate(queries)
  pdb.set_trace()
  pass
