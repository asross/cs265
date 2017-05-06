from lsmulator import *
from workloads import *
from bloom_assignments import *
from workloads import *
import scipy.stats
import numpy as np
import pickle

n = 50000
k = 5000


workloads = [
  UniformWorkload(n_queries=n, k_classes=k),
  #ZipfWorkload(n_queries=n, zipf_param=1.5),
]

i = 0
for workload in workloads:
  print(workload)
  for M, dM in [(500, 10)]:
    bl_trees = LSMulator.cache_vs_bloom_vs_buf_threaded(workload, M, dM=dM, balloc=baseline_assignment, verbose=True)
    for t in bl_trees:
      print(t.cache.size)
    #bl = [(t.memtbl.size, t.cache.size, t.disk_accesses) for t in bl_trees]
    i += 1
