from lsmulator import *
import numpy as np

# def compute_n_layers(workload, layer_size, layer_mult=2):
  # n_entries = len(set(workload))
  # result = 0
  # while n_entries > 0:
    # n_entries -= layer_size
    # result += 1
    # layer_size *= layer_mult
  # return n_layers

def optimize(initial_options, workload, iters):
  def perturb(opts):
    layer_size = opts['layer_size']
    cache_size = opts['cache_size']
    # bloom_size = opts['bloom_bit_length'] * opts['bloom_hash_count']
    # n_layers = compute_n_layers(workload, layer_size)
    # max_size = cache_size + layer_size + bloom_size * n_layers
    perturbed = dict(opts)
    layer_shift = np.random.randint(-layer_size//3, cache_size//3)
    perturbed['layer_size'] = layer_size + layer_shift
    perturbed['cache_size'] = cache_size - layer_shift
    # not quite right yet, doesn't account for bloom size
    return perturbed

  opts = dict(initial_options)
  cost = lsmulate(workload, **opts).disk_accesses()
  trace = [opts]
  costs = [cost]
  for t in range(iters):
    proposed_opts = perturb(opts)
    proposed_cost = lsmulate(workload, **proposed_opts).disk_accesses()
    if np.random.uniform() < (cost / proposed_cost)**(max(1,t/25.)):
      opts = perturb(opts)
      cost = proposed_cost
    trace.append(opts)
    costs.append(cost)

  return trace, costs
