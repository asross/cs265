from lsm_component import *
from collections import OrderedDict

def nth_last_slot(od, nth):
  i = 0
  for k, _ in od.items():
    if i == nth: return k
    i += 1

def last_slot(od):
  return nth_last_slot(od, 0)

class Cache(LSMComponent):
  def __init__(self, size):
    super(Cache, self).__init__(size)
    self.entries = OrderedDict()
    self.last_slot_hits = 0
    self.penultimate_hits = 0
    self.fake_last_slot = None
    self.fake_2ndl_slot = None

  def get(self, key):
    if self.size == 0:
      self.last_slot_hits += (key == self.fake_last_slot)
      self.penultimate_hits += (key == self.fake_2ndl_slot)
      return False

    if key in self.entries:
      self.hits += 1
      self.last_slot_hits += self.full and key == last_slot(self.entries)
      self.penultimate_hits += self.full and key == nth_last_slot(self.entries, 1)
      self.entries.move_to_end(key)
      return True
    else:
      self.misses += 1
      return False

  def put(self, key):
    if self.size == 0:
      self.fake_2ndl_slot = self.fake_last_slot
      self.fake_last_slot = key
      return

    if self.full:
      self.entries.popitem(last=False)
    self.entries[key] = None

  @property
  def keys(self):
    return list(reversed(self.entries.keys()))

if __name__ == '__main__':
  print('running cache tests...', end=' ')

  cache = Cache(3)

  # it's initially empty
  assert(cache.empty)
  assert(cache.keys == [])
  assert(last_slot(cache.entries) == None)

  # a get returns false, but adds the key...
  assert(not cache.get(5)); cache.put(5)
  assert(cache.keys == [5])
  assert(cache.get(5))
  assert(cache.keys == [5])
  assert(last_slot(cache.entries) == 5)
  assert(nth_last_slot(cache.entries, 1) == None)

  # ... to the _beginning_
  assert(not cache.get(4)); cache.put(4)
  assert(cache.keys == [4,5])
  assert(not cache.get(3)); cache.put(3)
  assert(cache.keys == [3,4,5])
  assert(cache.get(5))
  assert(cache.keys == [5,3,4])
  assert(last_slot(cache.entries) == 4)
  assert(nth_last_slot(cache.entries, 1) == 3)

  # when the cache is full, the last key is evicted
  assert(cache.full)
  assert(not cache.get(2)); cache.put(2)
  assert(cache.keys == [2,5,3])

  # it keeps stats
  assert(cache.hits == 2)
  assert(cache.misses == 4)
  assert(cache.hit_frequency == 1/3.)
  assert(cache.miss_frequency == 2/3.)

  cache.reset_counters()
  assert(cache.hits == 0)
  assert(cache.misses == 0)

  print('success!')
