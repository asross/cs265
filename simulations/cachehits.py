import numpy as np
from collections import Counter
import pdb

def cache_simulation(cache_size=50, n_queries=1000000, zipf_param=1.5):
  queries = np.random.zipf(zipf_param, n_queries)
  cache = list(range(1, cache_size+1))
  hits = Counter()
  misses = Counter()
  for query in queries:
    if query in cache:
      hits[query] += 1
      cache.remove(query)
    else:
      misses[query] += 1
      cache.remove(cache[-1])
    cache.insert(0, query)
  return hits, misses

hits, misses = cache_simulation()
pdb.set_trace()
pass
