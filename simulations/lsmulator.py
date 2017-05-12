import numpy as np
from cache import *
from layer import *
from bloom_assignments import *

default_layer_ratio = 2
default_memtbl_size = 100

def ceil(n): return int(np.ceil(n))

class LSMulator():
  def __init__(self, cache_size=50, memtbl_size=default_memtbl_size, layer_ratio=default_layer_ratio, bloom_size=100, page_size=256):
    self.cache = Cache(cache_size)
    self.memtbl = Layer(memtbl_size, ratio=layer_ratio, bsize=bloom_size, index=0)
    self.page_size = 256
    self.layer_queries = 0
    self.puts = 0
    self.gets = 0

  def clear_data(self):
      self.memtbl.entries = None
      for layer in self.layers:
          layer.entries = None
          if layer.bloom:
            layer.bloom.entries = None
      self.cache.entries = None

  def to_file(self, file_name):
    import dill
    with open(file_name, "wb") as f:
      return f.write(dill.dumps(self))

  @staticmethod
  def from_file(file_name):
    import dill
    with open(file_name, "rb") as f:
      return dill.loads(f.read())

  def put(self, key):
    self.puts += 1
    self.memtbl.put(key)

  def get(self, key):
    if key in self.memtbl.entries:
      self.memtbl.hits += 1
      self.memtbl.last_slot_hits += self.memtbl.full and key == self.memtbl.entries[-1]
      return True
    elif self.cache.get(key):
      return True
    else:
      self.layer_queries += 1
      result = self.memtbl.get(key)
      if result:
        self.cache.put(key)
      return result

  def reset_counters(self):
    for component in [self.cache, self.memtbl] + self.layers:
      component.reset_counters()

  def bigger_cache_savings(self, dM=1, p=1):
    return dM * (self.cache.last_slot_hits*p + self.cache.penultimate_hits*(1-p)) * (self.disk_accesses / self.layer_queries)

  def bigger_memtbl_savings(self, dM=1):
    return self.estimated_memtbl_read_savings(dM) #+ self.estimated_memtbl_write_savings(dM)

  def bigger_bloom_savings(self, dM=1, ballocs=monkey_assignment, bits_per_key=64):
    bc = np.array([l.bloom.bit_length for l in self.layers])

    total_bloom_bits = bc.sum()
    total_bloom_mem = int(round(total_bloom_bits / bits_per_key))

    b1 = ballocs(total_bloom_mem,    self.memtbl.size, self.layer_sizes, bits_per_key=bits_per_key)
    b2 = ballocs(total_bloom_mem+dM, self.memtbl.size, self.layer_sizes, bits_per_key=bits_per_key)

    if not np.allclose(b1, bc):
      print('Warning: current bloom allocation is different than', ballocs.__name__)

    das1 = sum([l.bloom.est_disk_accesses(m1) for m1, l in zip(b1, self.layers)])
    das2 = sum([l.bloom.est_disk_accesses(m2) for m2, l in zip(b2, self.layers)])

    return das1 - das2

  @property
  def layer_sizes(self):
    return np.array([l.size for l in self.layers])

  @property
  def layers(self):
    return self.memtbl.children()

  @property
  def disk_accesses(self):
    return sum(l.disk_accesses(self.page_size) for l in self.layers)

  @property
  def dupes_squashed(self):
    return sum(l.dupes_squashed for l in self.layers)

  @classmethod
  def emulate(kls, queries, **kwargs):
    lsmtree = kls(**kwargs)
    for key, is_write in queries:
      if is_write:
        lsmtree.put(key)
      else:
        lsmtree.get(key)
    return lsmtree

  @classmethod
  def cache_vs_bloom(kls, workload, total, dM=100, ballocs=monkey_assignment, layer_ratio=2, memtbl=default_memtbl_size):
    trees = []
    layers = LSMulator.emulate(workload.queries, memtbl_size=memtbl, layer_ratio=layer_ratio).layer_sizes
    for bloom in range(0, total, dM):
      trees.append(LSMulator.emulate(workload.queries,
        layer_ratio=layer_ratio,
        memtbl_size=memtbl,
        cache_size=total - bloom,
        bloom_size=ballocs(bloom, memtbl, layers)))
    return trees

  @classmethod
  def cache_vs_bloom_vs_buf(kls, workload, total, dM=100, ballocs=monkey_assignment, layer_ratio=2, verbose=False):
    trees = []
    for memtbl in range(dM, total + dM, dM):
      if verbose:
        print('Memtable =', memtbl)
      layers = LSMulator.emulate(workload.queries, memtbl_size=memtbl, layer_ratio=layer_ratio).layer_sizes
      for bloom in range(0, total - memtbl + dM, dM):
        trees.append(LSMulator.emulate(workload.queries,
          layer_ratio=layer_ratio,
          memtbl_size=memtbl,
          cache_size=total - memtbl - bloom,
          bloom_size=ballocs(bloom, memtbl, layers)))
    return trees

  @staticmethod
  def cache_vs_bloom_vs_buf_threaded(workload, total, dM=100, balloc=monkey_assignment, layer_ratio=2, verbose=False):
    import pathos.multiprocessing as mp
    pool = mp.ProcessingPool(nodes=4)

    def get_layer_size_wrapper(memtbl):
      return list(LSMulator.emulate(
        workload.queries,
        memtbl_size=memtbl,
        layer_ratio=layer_ratio).layer_sizes)

    layer_sizes = list(pool.map(get_layer_size_wrapper, range(dM, total, dM)))

    # Pre-generate jobs
    def generate_jobs():
      for i, memtbl in enumerate(range(dM, total + dM, dM)):
        layers = layer_sizes[i]
        for bloom in range(0, total - memtbl + dM, dM):
          yield dict(
              queries=workload.queries,
              layer_ratio=layer_ratio,
              memtbl_size=memtbl,
              cache_size=total - memtbl - bloom,
              bloom_size=balloc(bloom, memtbl, layers))

    def wrapper(args):
      queries = args.pop("queries")
      return LSMulator.emulate(queries, **args)

    results = pool.uimap(wrapper, generate_jobs())

    return results

  def estimated_memtbl_read_savings(self, dM=1):
    return self.memtbl.read_savings(dM)

  def estimated_memtbl_write_savings(tree, dM=1, check_tree=None, godexists=False):
    accesses_1 = 0
    accesses_2 = 0
    ##get the entries that would have ended up in tree2

    tree2_memtbl_size = tree.memtbl.size + dM
    tree2_layer_sizes = [tree.memtbl.size + dM] + [(tree.memtbl.size + dM)*2**(i+1) for i in range(len(tree.layers))]
    remaining_entries = np.sum([len(tree.layers[i].entries) for i in range(len(tree.layers))]) + len(tree.memtbl.entries)
    layer_total_capacity = [np.sum(tree2_layer_sizes[:i]) for i in range(len(tree.layers)+1,0,-1)][::-1]
    layer_entries = np.zeros((len(tree.layers)+1))
    for i in range(len(tree.layers),0,-1):
      if remaining_entries> layer_total_capacity[i-1]:
        if (remaining_entries - layer_total_capacity[i-1])/(layer_total_capacity[i-1]-layer_total_capacity[i-2])==2:
          layer_entries[i] = (ceil((remaining_entries - layer_total_capacity[i-1])/(layer_total_capacity[i-1]-layer_total_capacity[i-2]))-1)*tree2_layer_sizes[i-1]
        else:
          layer_entries[i] = ceil((remaining_entries - layer_total_capacity[i-1])/(layer_total_capacity[i-1]-layer_total_capacity[i-2]))*tree2_layer_sizes[i-1]
        remaining_entries=remaining_entries - layer_entries[i]
    layer_entries[0] = remaining_entries%tree2_layer_sizes[0]
    remaining_entries = remaining_entries - layer_entries[0]
    layer_entries[1] = remaining_entries
    layer_entries = layer_entries[1:]
    
    tree2_layer_sizes=tree2_layer_sizes[1:]
    tree2_n_in = []
    tree2_n_in.append((tree.memtbl.n_in // tree2_memtbl_size)*tree2_memtbl_size)
    for i in range(len(tree.layers)-1):
      if (tree2_n_in[-1]%tree2_layer_sizes[i])!=0:
        tree2_n_in.append(((tree2_n_in[-1] // tree2_layer_sizes[i])*tree2_layer_sizes[i])-tree.layers[i].dupes_squashed)
      else:
        tree2_n_in.append((((tree2_n_in[-1] // tree2_layer_sizes[i])-1)*tree2_layer_sizes[i])-tree.layers[i].dupes_squashed)


    write_accs_1 = ceil(tree.memtbl.size/256) * ceil(tree.layers[0].n_in/tree.memtbl.size/2) \
    + (ceil(tree.memtbl.size*2/256) *((tree.layers[0].n_in/tree.memtbl.size)//2))
    
    write_accs_2 = ceil(tree2_memtbl_size/256) * ceil(tree2_n_in[0]/tree2_memtbl_size/2) \
    + (ceil(tree2_memtbl_size*2/256) *((tree2_n_in[0]/tree2_memtbl_size)//2))
    
    read_accs_1 = np.round((ceil(tree.layers[1].n_in/tree.layers[0].size) * (ceil(tree.layers[0].size/256) + ceil(tree.layers[0].size/2/256)) \
    + (len(tree.layers[0].entries)>tree.memtbl.size) * ceil(tree.memtbl.size/256))*1/(1-(tree.layers[0].dupes_squashed/float(tree.layers[0].n_in))))
    
    read_accs_2 = ceil(tree2_n_in[1]/tree2_layer_sizes[0]) * (ceil(tree2_layer_sizes[0]/256) + ceil(tree2_layer_sizes[0]/2/256)) \
    + (layer_entries[0]>tree2_memtbl_size) * ceil(tree2_memtbl_size/256)
    
    accesses_1 += write_accs_1
    accesses_2 += write_accs_2
    
    accesses_1 += read_accs_1
    accesses_2 += read_accs_2
    
    if godexists:
      print(tree2_n_in)
      print([check_tree.layers[i].n_in for i in range(len(check_tree.layers))])
      
      print("pred writes", write_accs_1)
      print("true writes", tree.layers[0].write_accesses())
      
      print("pred writes 2", write_accs_2)
      print("true writes 2", check_tree.layers[0].write_accesses())
      
      print("pred reads", read_accs_1)
      print("true reads", tree.layers[0].read_accesses())
      
      print("pred reads 2", read_accs_2)
      print("true reads 2", check_tree.layers[0].read_accesses())
      
    #for all other layers:
    for i in range(1,len(tree.layers)-1):
      layer_writes_1 = ceil(tree.layers[i-1].size/256) * ceil(tree.layers[i].n_in/tree.layers[i-1].size/2) \
      + (ceil(tree.layers[i-1].size*2*(1-tree.layers[i].dupes_squashed/float(tree.layers[i].n_in))/256) *((tree.layers[i].n_in/tree.layers[i-1].size)//2))
      
      layer_writes_2 = ceil(tree2_layer_sizes[i-1]/256) * ceil(tree2_n_in[i]/tree2_layer_sizes[i-1]/2) \
      + (ceil(tree2_layer_sizes[i-1]*2*(1-tree.layers[i].dupes_squashed/float(tree.layers[i].n_in))/256) *((tree2_n_in[i]/tree2_layer_sizes[i-1])//2))
      
      layer_reads_1 = (ceil(tree.layers[i+1].n_in/tree.layers[i].size) * (ceil(tree.layers[i].size/256) + ceil(tree.layers[i].size/2/256)) \
      + (len(tree.layers[i].entries)>tree.layers[i-1].size) * ceil(tree.layers[i-1].size/256))*(1/(1-tree.layers[i].dupes_squashed/float(tree.layers[i].n_in)))
      
      layer_reads_2 = ceil(tree2_n_in[i+1]/tree2_layer_sizes[i]) * (ceil(tree2_layer_sizes[i]/256) + ceil(tree2_layer_sizes[i]/2/256)) \
      + (layer_entries[i]>tree2_layer_sizes[i-1]) * ceil(tree2_layer_sizes[i-1]/256)
      
      accesses_1 += layer_writes_1
      accesses_2 += layer_writes_2
      
      accesses_1 += layer_reads_1
      accesses_2 += layer_reads_2
      
      if godexists:
        print("Layer {}".format(i-1))
        
        print("layer", i-1, "pred writes", layer_writes_1)
        print("layer", i-1, "true writes", tree.layers[i].write_accesses())
        
        print("layer", i-1, "pred writes 2", layer_writes_2)
        print("layer", i-1, "true writes 2", check_tree.layers[i].write_accesses())
        
        print("layer", i-1, "pred reads", layer_reads_1)
        print("layer", i-1, "true reads", tree.layers[i].read_accesses())
        
        print("layer", i-1, "pred reads 2", layer_reads_2)
        print("layer", i-1, "true reads 2", check_tree.layers[i].read_accesses())      
   
    if godexists:
      print(accesses_1-accesses_2)
      print(np.sum([tree.layers[i].read_accesses() + tree.layers[i].write_accesses() for i in range(len(tree.layers))]))
      print(np.sum([check_tree.layers[i].read_accesses() + check_tree.layers[i].write_accesses() for i in range(len(check_tree.layers))]))
      print(np.sum([tree.layers[i].read_accesses() + tree.layers[i].write_accesses() for i in range(len(tree.layers))]) - \
         np.sum([check_tree.layers[i].read_accesses() + check_tree.layers[i].write_accesses() for i in range(len(check_tree.layers))]))
    
    return accesses_1-accesses_2




if __name__ == '__main__':
  import pdb
  from workloads import readwritify
  queries = readwritify(np.random.zipf(1.5, 100000), update_fraction=0.05, null_read_fraction=0.01)
  lsmtree = LSMulator.emulate(queries, bloom_size=4096)
  pdb.set_trace()
  pass
