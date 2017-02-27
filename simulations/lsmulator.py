import numpy as np
from collections import Counter, defaultdict
import pdb

class SimulatedComponent():
  def __init__(self, size):
    self.hits = Counter()
    self.misses = Counter()
    self.size = size
    self.entries = []
  def is_full(self): return len(self.entries) == self.size
  def remaining_capacity(self): return self.size - len(self.entries)
  def total_hits(self): return sum(self.hits.values())
  def total_misses(self): return sum(self.misses.values())
  def total_accesses(self): return self.total_hits() + self.total_misses()
  def hit_frequency(self): return self.total_hits() / float(self.total_accesses())
  def miss_frequency(self): return self.total_misses() / float(self.total_accesses())

class SimulatedCache(SimulatedComponent):
  def get(self, key):
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

class SimulatedBloomFilter():
  def __init__(self, bit_length=100, hash_count=10):
    self.bit_length = bit_length
    self.hash_count = hash_count
    self.reset()

  def reset(self):
    self.entries = []
    self.collisions = defaultdict(lambda: defaultdict(lambda: None))

  @property
  def size(self):
    return self.bit_length * self.hash_count

  @property
  def missprob(self):
    return (1./self.bit_length)**self.hash_count

  def put(self, key):
    self.entries.append(key)

  def false_positive(self, k1):
    for k2 in self.entries:
      if self.collisions[k2][k1] is None: self.collisions[k2][k1] = (np.random.random() < self.missprob)
      if self.collisions[k2][k1]: return True
    return False

  def get(self, key):
    if key in self.entries:
      return True
    else:
      return self.false_positive(key)

class SimulatedLayer(SimulatedComponent):
  def __init__(self, size, ratio=2, bloom_bit_length=10, bloom_hash_count=4):
    super(SimulatedLayer, self).__init__(size)
    self.bloom_filter = SimulatedBloomFilter(bit_length=bloom_bit_length, hash_count=bloom_hash_count)
    self.next_layer = None
    self.ratio = ratio

  def put(self, key):
    if self.is_full(): self.merge_down()
    self.entries.append(key)
    self.bloom_filter.put(key)

  def merge(self, entries):
    assert(len(entries) <= self.size)
    if len(entries) > self.remaining_capacity(): self.merge_down()
    self.entries += entries
    for key in entries:
      self.bloom_filter.put(key)

  def get(self, key):
    if key in self.entries:
      self.hits[key] += 1
      return True
    else:
      if self.bloom_filter.false_positive(key):
        self.misses[key] += 1
      if self.next_layer is None:
        return False
      else:
        return self.next_layer.get(key)

  def merge_down(self):
    # if we can't accept all the new entries, then merge existing entries to the next layer
    if self.next_layer is None: # creating it if it does not exist
      self.next_layer = SimulatedLayer(self.size * self.ratio, self.ratio)
    self.next_layer.merge(self.entries)
    self.entries = []
    self.bloom_filter.reset()

  def self_and_children(self):
    if self.next_layer is None:
      return [self]
    else:
      return [self] + self.next_layer.self_and_children()

class LSMulator():
  def __init__(self, cache_size=50, layer_size=100, layer_ratio=2, bloom_bit_length=100, bloom_hash_count=10):
    self.cache = SimulatedCache(cache_size)
    self.top_layer = SimulatedLayer(layer_size,
        ratio=layer_ratio, bloom_bit_length=bloom_bit_length, bloom_hash_count=bloom_hash_count)

  def put(self, key):
    self.top_layer.put(key)

  def get(self, key):
    return True if self.cache.get(key) else self.top_layer.get(key)

  def layers(self):
    return self.top_layer.self_and_children()

def lsmulate(n_queries=100000, cache_size=50, layer_size=100, query_generator=lambda n: np.random.zipf(1.5, n)):
  queries = query_generator(n_queries)
  lsmtree = LSMulator(cache_size=cache_size, layer_size=layer_size)
  entries = set()
  for key in queries:
    if key in entries:
      lsmtree.get(key)
    else:
      lsmtree.put(key)
      entries.add(key)
  return lsmtree, entries

if __name__ == '__main__':
  tree, all_entries = lsmulate()
  pdb.set_trace()
  pass
