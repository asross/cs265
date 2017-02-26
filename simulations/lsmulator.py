import numpy as np
from collections import Counter
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
  def miss_frequency(self): return self.total_hits() / float(self.total_accesses())

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

class SimulatedLayer(SimulatedComponent):
  def __init__(self, size, ratio=2):
    super(SimulatedLayer, self).__init__(size)
    self.next_layer = None
    self.ratio = ratio

  def put(self, key):
    if self.is_full(): self.merge_down()
    self.entries.append(key)

  def merge(self, entries):
    assert(len(entries) <= self.size)
    if len(entries) > self.remaining_capacity(): self.merge_down()
    self.entries += entries

  def get(self, key):
    if key in self.entries:
      self.hits[key] += 1
      return True
    else:
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

  def self_and_children(self):
    if self.next_layer is None:
      return [self]
    else:
      return [self] + self.next_layer.self_and_children()

class LSMulator():
  def __init__(self, cache_size=50, layer_size=100):
    self.cache = SimulatedCache(cache_size)
    self.layer = SimulatedLayer(layer_size)

  def put(self, key):
    self.layer.put(key)

  def get(self, key):
    return True if self.cache.get(key) else self.layer.get(key)

  def layers(self):
    return self.layer.self_and_children()

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
