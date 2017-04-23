from lsm_component import *
from bloom import *
import numpy as np

class Layer(LSMComponent):
  def __init__(self, size, ratio=2, index=0, bsize=100):
    super(Layer, self).__init__(size)
    self.child = None
    self.ratio = ratio
    self.index = index
    self.bsize = bsize
    self.bloom = Bloom(size, bsize, index) if index else None
    self.mergedowns = 0
    if index > 0:
      self.entries = set()

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
    self.entries.update(entries)
    for key in entries:
      self.bloom.put(key)

  def get(self, key):
    if key in self.entries:
      self.hits += 1
      return True
    else:
      if self.bloom is None or self.bloom.get(key):
        self.misses += 1
      if self.child is None:
        return False
      else:
        return self.child.get(key)

  def merge_down(self):
    self.mergedowns += 1
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

  assert(layer.mergedowns == 6)
  assert(layer.child.mergedowns == 2)
  assert(layer.child.child.mergedowns == 0)
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

