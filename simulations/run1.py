from lsmulator import *
from workloads import *
from bloom_assignments import *
from workloads import *
import scipy.stats
import numpy as np
import pickle

n = 100000
k = 10000

workloads = [
  UniformWorkload(n_queries=n, k_classes=k),
  RoundRobinWorkload(n_queries=n, k_classes=k),
  ZipfWorkload(n_queries=n, zipf_param=1.1),
  ZipfWorkload(n_queries=n, zipf_param=1.5),
  DiscoverDecayWorkload(n_queries=n),
  DiscoverDecayWorkload(n_queries=n, decay_rate=scipy.stats.beta(2, 1)),
  DiscoverDecayWorkload(n_queries=n, updates=scipy.stats.poisson(16), decay_rate=scipy.stats.beta(2, 1)),
  PeriodicDecayWorkload(n_queries=n, period=100),
  PeriodicDecayWorkload(n_queries=n, period=1000),
  PeriodicDecayWorkload(n_queries=n, period=1000, decay_rate=scipy.stats.beta(2, 1))
]

i = 0
for workload in workloads:
  print(workload)
  for M, dM in [(1000, 50), (10000, 200), (50000, 500)]:
    bl_trees = LSMulator.cache_vs_bloom_vs_buf(workload, M, dM=dM, balloc=baseline_assignment, verbose=True)
    mk_trees = LSMulator.cache_vs_bloom_vs_buf(workload, M, dM=dM, balloc=monkey_assignment, verbose=True)
    with open('./results{}.pkl'.format(i), 'wb') as f:
      pickle.dump((workload, M, dM, bl_trees, mk_trees), f)
    i += 1
