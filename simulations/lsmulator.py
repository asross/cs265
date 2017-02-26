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

class LSMulator():
  def __init__(self, cache_size=50, layer_size=100):
    self.cache = SimulatedCache(cache_size)
    self.layer = SimulatedLayer(layer_size)

  def put(self, key):
    self.layer.put(key)

  def get(self, key):
    if self.cache.get(key):
      return
    self.layer.get(key)

  def layers(self):
    layer = self.layer
    layers = [layer]
    while layer.next_layer:
      layer = layer.next_layer
      layers.append(layer)
    return layers

def lsmulate(n_queries=100000, cache_size=50, layer_size=100, zipf_param=1.5):
  queries = np.random.zipf(zipf_param, n_queries)
  lsm_tree = LSMulator(cache_size=cache_size, layer_size=layer_size)
  all_entries = set()
  for key in queries:
    if key in all_entries:
      lsm_tree.get(key)
    else:
      lsm_tree.put(key)
      all_entries.add(key)
  return lsm_tree

if __name__ == '__main__':
  tree = lsmulate()
  pdb.set_trace()
  pass
