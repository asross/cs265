class Optimizer():
  def __init__(self, workload, max_size):
    pass

# Rough idea here is, given a workload and a cost model,
# can we tweak LSM tree cache/layer/bloom sizes such
# that we arrive at the lowest cost?
#
# Maybe we also want to optimize for a set of workloads,
# maybe with importance weights.
