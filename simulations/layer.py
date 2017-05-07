from lsm_component import *
from bloom import *
import numpy as np
from collections import defaultdict

class Layer(LSMComponent):
  def __init__(self, size, ratio=2, index=0, bsize=100):
    super(Layer, self).__init__(size)
    self.child = None
    self.ratio = ratio
    self.index = index
    self.bsize = bsize
    self.mergereads = []
    self.mergewrites = []
    self.dupes_squashed = 0
    if index > 0:
      self.entries = set()
      self.bloom = Bloom(size, bsize, index)
    else:
      self.hit_indexes = defaultdict(int)
      self.bloom = None

  def reset_counters(self):
    super(Layer, self).reset_counters()
    self.mergereads = []
    self.mergewrites = []
    if self.bloom:
      self.bloom.reset_counters()

  def put(self, key):
    if self.full:
      self.merge_down()

    if self.index:
      self.entries.add(key)
      self.bloom.put(key)
    else:
      self.entries.append(key)

  def merge(self, entries):
    assert(len(entries) <= self.size)

    if len(entries) > self.free_space:
      self.merge_down()

    if len(self.entries):
      self.mergereads.append(len(self.entries))

    pre_merge_length = len(self.entries) + len(entries)
    self.entries.update(entries)
    post_merge_length = len(self.entries)

    for key in entries:
      self.bloom.put(key)

    self.mergewrites.append(post_merge_length)
    self.dupes_squashed += pre_merge_length - post_merge_length

  def get(self, key):
    if self.index == 0:
      # MEMTABLE
      try:
        i = self.entries.index(key)
        self.hits += 1
        self.hit_indexes[i] += 1
        return True
      except ValueError:
        self.misses += 1
    else:
      # LAYER
      if key in self.entries:
        self.hits += 1
        return True
      elif self.bloom.get(key):
        self.misses += 1

    # If we didn't return, check the child
    if self.child is None:
      return False
    else:
      return self.child.get(key)

  def disk_accesses(self, pagesize=256):
    total = self.accesses
    for n_entries in self.mergereads + self.mergewrites:
      total += int(np.ceil(n_entries / float(pagesize)))
    return total

  def merge_down(self):
    if self.index > 0:
      self.mergereads.append(len(self.entries))
    # if we can't accept all the new entries, then merge existing entries to the next layer
    if self.child is None: # creating it if it does not exist
      self.child = Layer(self.size*self.ratio, self.ratio, bsize=self.bsize, index=self.index+1)
    self.child.merge(self.entries)
    if self.bloom:
      self.entries = set()
      self.bloom.reset()
    else:
      self.entries = []

  def children(self):
    layers = []
    layer = self.child
    while layer is not None:
      layers.append(layer)
      layer = layer.child
    return layers

def layer_sizes(n, memtbl_size, layer_ratio):
  n -= memtbl_size
  size = memtbl_size
  sizes = []
  while n > 0:
    size *= layer_ratio
    sizes.append(size)
    n -= size
  return np.array(sizes)

if __name__ == '__main__':
  print('running layer tests...', end=' ')
  np.testing.assert_array_equal(layer_sizes(100, 16, 2), [32, 64])
  layer = Layer(16, 2)
  assert(layer.child == None)
  assert(layer.bloom == None)

  for i in range(100):
    layer.put(i)
  np.testing.assert_array_equal(layer_sizes(100, 16, 2), [l.size for l in layer.children()])

  assert(sum(layer.mergereads) == 0)
  assert(layer.disk_accesses() == 0)

  assert(sum(layer.child.mergereads) == 112)
  assert(sum(layer.child.mergewrites) == 144)

  assert(layer.child.disk_accesses(pagesize=16) == 16)
  assert(layer.child.disk_accesses(pagesize=256) == 11)
  assert(layer.child.disk_accesses(pagesize=7) == 43)

  assert(sum(layer.child.child.mergereads) == 32)
  assert(sum(layer.child.child.mergewrites) == 96)

  assert(layer.child.child.child == None)

  assert(layer.child.bloom is not None)
  assert(layer.child.child.bloom is not None)

  assert(layer.entries == [96, 97, 98, 99])
  assert(layer.child.entries == set(range(64, 96)))
  assert(layer.child.child.entries == set(range(64)))

  assert(layer.get(96))
  assert(layer.hits == 1)
  assert(layer.misses == 0)

  assert(layer.get(95))
  assert(layer.hits == 1)
  assert(layer.misses == 1)
  assert(layer.child.hits == 1)
  assert(layer.child.misses == 0)

  assert(not layer.get(100))
  assert(not (layer.child.misses and layer.child.child.misses)) # only sometimes true

  print('success!')

