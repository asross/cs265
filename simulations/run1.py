from lsmulator import *
from workloads import *
from bloom_assignments import *
from workloads import *
import scipy.stats
import numpy as np
import dill as pickle
import traceback

n = 50000
k = 5000

workloads = [
  UniformWorkload(n_queries=n, k_classes=k),
  RoundRobinWorkload(n_queries=n, k_classes=k),
  ZipfWorkload(n_queries=n, zipf_param=1.1),
  ZipfWorkload(n_queries=n, zipf_param=1.5),
  ZipfWorkload(n_queries=n, zipf_param=1.8),
  EightyTwentyWorkload(n_queries=n, k_classes=k),
  EightyTwentyWorkload(n_queries=n, k_classes=k, update_fraction=0.25),
  EightyTwentyWorkload(n_queries=n, k_classes=k, update_fraction=0.65),
  MultinomialWorkload(n_queries=n, k_classes=k),
  MultinomialWorkload(n_queries=n, k_classes=k, dist=scipy.stats.gamma(2)),
  MultinomialWorkload(n_queries=n, k_classes=k, dist=scipy.stats.expon(1)),
  DiscoverDecayWorkload(n_queries=n),
  DiscoverDecayWorkload(n_queries=n, lookups=scipy.stats.poisson(8)),
  DiscoverDecayWorkload(n_queries=n, decay_rate=scipy.stats.beta(2, 1)),
  DiscoverDecayWorkload(n_queries=n, updates=scipy.stats.poisson(0), decay_rate=scipy.stats.beta(2, 1)),
  DiscoverDecayWorkload(n_queries=n, updates=scipy.stats.poisson(16), decay_rate=scipy.stats.beta(2, 1)),
  PeriodicDecayWorkload(n_queries=n, period=100),
  PeriodicDecayWorkload(n_queries=n, period=1000),
  PeriodicDecayWorkload(n_queries=n, period=1000, decay_rate=scipy.stats.beta(2, 1))
]

i = 0
for workload in workloads:
  print(workload)
  for M, dM in [(1000, 50)]:
    print(M, dM)

    try:
      print("Baseline.")
      bl_trees = LSMulator.cache_vs_bloom_vs_buf_threaded(workload, M, dM=dM, balloc=baseline_assignment, verbose=True)
      bl = list(bl_trees)
      for t in bl:
        t.clear_data()
      with open('./results-{}-{}-{}-{}.pkl'.format(str(workload), M, dM, "bl"), 'wb') as f:
        pickle.dump((str(workload), M, dM, bl), f)
    except Exception as ex:
      traceback.print_exc()

    try:
      print("Monkey.")
      mk_trees = LSMulator.cache_vs_bloom_vs_buf_threaded(workload, M, dM=dM, balloc=monkey_assignment, verbose=True)
      mk = list(mk_trees)
      for t in mk:
        t.clear_data()
      with open('./results-{}-{}-{}-{}.pkl'.format(str(workload), M, dM, "mk"), 'wb') as f:
        pickle.dump((str(workload), M, dM, mk), f)
    except Exception as ex:
      traceback.print_exc()

    i += 1
