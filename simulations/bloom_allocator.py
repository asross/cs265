import numpy as np

ln2 = np.log(2)
ln22 = ln2 * ln2

def ceil(x): return int(np.ceil(x))

def baseline_assignment(max_memory, lsmtree):
  memtbl_size = lsmtree.memtbl.size
  layer_sizes = lsmtree.layer_sizes

  diff = memtbl_size
  assignment = np.ones_like(layer_sizes) * (max_memory / len(layer_sizes))
  calc_R = lambda i: np.exp(-(assignment[i]/layer_sizes[i]) * ln22)
  diff_R = lambda i, j: np.abs(calc_R(i) - calc_R(j))

  while diff > 1:
    change = False
    for i in range(0, len(layer_sizes)-1):
      for j in range(i+1, len(layer_sizes)):
        diff_orig = diff_R(i,j)
        assignment[i] += diff
        assignment[j] -= diff
        if diff_R(i,j) < diff_orig and assignment[i] > 0 and assignment[j] > 0:
          change = True
          continue
        assignment[i] -= diff * 2
        assignment[j] += diff * 2
        if diff_R(i,j) < diff_orig and assignment[i] > 0 and assignment[j] > 0:
          change = True
          continue
        assignment[i] += diff
        assignment[j] -= diff
    if not change:
      diff /= 2

  return assignment

def monkey_assignment(max_memory, lsmtree):
  memtbl_size = lsmtree.memtbl.size
  layer_sizes = lsmtree.layer_sizes

  diff = memtbl_size
  assignment = np.ones_like(layer_sizes) * (max_memory / len(layer_sizes))
  eval_R = lambda: sum(np.exp(-(m/n) * ln22) for m,n in zip(assignment, layer_sizes))
  curr_R = eval_R()

  while diff > 1:
    change = False
    for i in range(0, len(layer_sizes)-1):
      for j in range(i+1, len(layer_sizes)):
        assignment[i] += diff
        assignment[j] -= diff
        value = eval_R()
        if value < curr_R and value > 0 and assignment[j] > 0:
          curr_R = value
          change = True
          continue
        assignment[i] -= diff * 2
        assignment[j] += diff * 2
        value = eval_R()
        if value < curr_R and value > 0 and assignment[i] > 0:
          curr_R = value
          change = True
          continue
        assignment[i] += diff
        assignment[j] -= diff
    if not change:
      diff /= 2

  return assignment
