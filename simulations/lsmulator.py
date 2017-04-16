import numpy as np
from collections import Counter, defaultdict
import pdb

class LSMComponent():
  def __init__(self, size):
    self.hits = Counter()
    self.misses = Counter()
    self.size = size
    self.entries = []
  def is_full(self): return len(self.entries) == self.size
  def free_space(self): return self.size - len(self.entries)
  def total_hits(self): return sum(self.hits.values())
  def total_misses(self): return sum(self.misses.values())
  def total_accesses(self): return self.total_hits() + self.total_misses()
  def hit_frequency(self): return self.total_hits() / float(self.total_accesses())
  def miss_frequency(self): return self.total_misses() / float(self.total_accesses())

class Cache(LSMComponent):
  def get(self, key):
    if self.size == 0: return False

    if key in self.entries:
      self.hits[key] += 1
      result = True
      if self.is_full(): self.entries.remove(key)
    else:
      self.misses[key] += 1
      result = False
      if self.is_full(): self.entries.remove(self.entries[-1])
    self.entries.insert(0, key)
    return result

class Layer(LSMComponent):
  def __init__(self, size, ratio=2, index=0, bsize=100):
    super(Layer, self).__init__(size)
    self.child = None
    self.ratio = ratio
    self.index = index
    self.bsize = bsize
    self.bloom = BloomFilter(size, bsize, index) if index else None

  def put(self, key):
    if self.is_full():
      self.merge_down()
    self.entries.append(key)
    if self.bloom is not None:
      self.bloom.put(key)

  def merge(self, entries):
    assert(len(entries) <= self.size)
    if len(entries) > self.free_space(): self.merge_down()
    self.entries += entries
    if self.bloom is not None:
      for key in entries:
        self.bloom.put(key)

  def get(self, key):
    if key in self.entries:
      self.hits[key] += 1
      return True
    else:
      if self.bloom and self.bloom.get(key):
        self.misses[key] += 1
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

class BloomFilter():
  def __init__(self, size, bit_length=100, index=0):
    if callable(bit_length):
      bit_length = bit_length(index)
    elif isinstance(bit_length, (list, tuple, np.ndarray)):
      bit_length = bit_length[index-1]
    self.layer_size = size # n
    self.bit_length = bit_length # m
    self.hash_count = int(np.ceil((bit_length / size) * np.log(2))) # k
    self.reset()

  def reset(self):
    self.entries = set()
    self.hashes = defaultdict(self.random_hash_eval)

  def random_hash_eval(self):
    return tuple(np.random.choice(self.bit_length) for _ in range(self.hash_count))

  def put(self, key):
    self.entries.update(self.hashes[key])

  def get(self, key):
    return self.entries.issuperset(self.hashes[key])

  @property
  def size(self):
    return self.bit_length * self.hash_count

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
