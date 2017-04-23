from lsm_component import *

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

if __name__ == '__main__':
  print('running cache tests...', end=' ')

  cache = Cache(3)

  # it's initially empty
  assert(cache.empty)
  assert(cache.entries == [])

  # a get returns false, but adds the key...
  assert(not cache.get(5))
  assert(cache.entries == [5])
  assert(cache.get(5))
  assert(cache.entries == [5])

  # ... to the _beginning_
  assert(not cache.get(4))
  assert(cache.entries == [4,5])
  assert(not cache.get(3))
  assert(cache.entries == [3,4,5])
  assert(cache.get(5))
  assert(cache.entries == [5,3,4])

  # when the cache is full, the last key is evicted
  assert(cache.full)
  assert(not cache.get(2))
  assert(cache.entries == [2,5,3])

  # it keeps stats
  assert(cache.hits == 2)
  assert(cache.misses == 4)
  assert(cache.hit_frequency == 1/3.)
  assert(cache.miss_frequency == 2/3.)

  print('success!')
