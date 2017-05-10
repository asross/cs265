from collections import defaultdict
from lsm_component import *
import numpy as np

class Bloom(LSMComponent):
  def __init__(self, size, bit_length=100, index=0):
    super(Bloom, self).__init__(size)
    if callable(bit_length):
      bit_length = bit_length(index)
    elif isinstance(bit_length, (list, tuple, np.ndarray)):
      bit_length = bit_length[index-1]
    self.bit_length = bit_length # m
    self.hash_count = int(np.ceil((bit_length / size) * np.log(2))) # k
    self.lengths = defaultdict(int)
    self.reset()

  def reset(self):
    self.entries = set()
    self.hashes = set()
    self.hash_for = defaultdict(self.random_hash_eval)

  def random_hash_eval(self):
    if self.hash_count == 0 or self.bit_length == 0:
      return tuple()

    return tuple(np.floor(np.random.uniform(size=self.hash_count)*self.bit_length).astype(int))

  def put(self, key):
    self.entries.add(key)
    self.hashes.update(self.hash_for[key])

  def get(self, key):
    self.lengths[len(self.entries)] += 1
    result = self.hashes.issuperset(self.hash_for[key])
    if key not in self.entries:
      if result:
        self.misses += 1
      else:
        self.hits += 1
    return result

  @property
  def false_positive_rate(self):
    return self.miss_frequency

  def estimated_fp_rate(self, m=None):
    if m is None:
      m = self.bit_length
    total = float(sum(self.lengths.values()))
    return sum([(count/total)*(0.6185**(m/float(n))) for n, count in self.lengths.items()])

  def est_disk_accesses(self, m=None):
    return self.estimated_fp_rate(m) * self.accesses

if __name__ == '__main__':
  print('running bloom tests...', end=' ')
  bf = Bloom(100, 1000, 0)
  assert(bf.hash_count == 7)

  # an empty bloom filter returns false (++hits)
  assert(not bf.get(5))
  assert(bf.hits == 1)
  assert(bf.misses == 0)

  # we can fill it
  bf.put(5)
  assert(len(bf.hash_for) == 1)
  assert(list(bf.entries) == [5])

  # getting entries in the filter is fine
  assert(bf.get(5))
  assert(len(bf.hash_for) == 1)
  assert(list(bf.entries) == [5])
  assert(bf.hits == 1)
  assert(bf.misses == 0)

  # if we force a hash collision...
  bf.hash_for[7] = bf.hash_for[5]

  # then we incorrectly return false (and ++misses)
  assert(bf.get(7))
  assert(bf.hits == 1)
  assert(bf.misses == 1)
  print('success!')
