import numpy as np

def baseline_assignment(max_memory_in_keys, memtbl_key_count, layer_key_counts, bits_per_key=64):
  if len(layer_key_counts) == 0:
    return np.array([])
  bits = max_memory_in_keys * bits_per_key
  diff = memtbl_key_count * bits_per_key
  assignment = np.ones_like(layer_key_counts) * (bits / len(layer_key_counts))

  calc_R = lambda i: np.exp(-(assignment[i]/layer_key_counts[i]) * np.log(2) * np.log(2))
  diff_R = lambda i, j: np.abs(calc_R(i) - calc_R(j))

  while diff > 1:
    change = False
    for i in range(0, len(layer_key_counts)-1):
      for j in range(i+1, len(layer_key_counts)):
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

  return assignment.astype(int)

def monkey_assignment(max_memory_in_keys, memtbl_key_count, layer_key_counts, bits_per_key=64):
  if len(layer_key_counts) == 0:
    return np.array([])
  bits = max_memory_in_keys * bits_per_key
  diff = memtbl_key_count * bits_per_key
  assignment = np.ones_like(layer_key_counts) * (bits / len(layer_key_counts))

  eval_R = lambda: sum(np.exp(-(m/float(n)) * np.log(2) * np.log(2)) for m,n in zip(assignment, layer_key_counts))
  curr_R = eval_R()

  while diff > 1:
    change = False
    for i in range(0, len(layer_key_counts)-1):
      for j in range(i+1, len(layer_key_counts)):
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

  return assignment.astype(int)

if __name__ == '__main__':
  print('running bloom allocation tests...', end=' ')
  # On https://sigmodsubmitter.github.io, this corresponds to selecting:
  # #-Entries: 10,000,000
  # Entry size: 4
  # Buffer size: 1
  # Bloom size: 1
  # Size ratio: 2 (leveling)
  memtbl = 262144
  layers = np.array([ 524288, 1048576, 2097152, 4194304, 2135680])
  malloc = monkey_assignment(memtbl, memtbl, layers, bits_per_key=32)
  assert(list(malloc) == [2004397, 2496021, 1966491, 1, 1921695])
  balloc = baseline_assignment(memtbl, memtbl, layers, bits_per_key=32)
  assert(list(balloc) == [439805,  879609, 1759217, 3518437, 1791537])
  print('success!')
