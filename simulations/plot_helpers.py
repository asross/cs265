import matplotlib.pyplot as plt
from figure_grid import *
from workloads import *

def plot_access_piechart(lsmtree, title=None):
  memtbl = lsmtree.memtbl
  cache = lsmtree.cache
  layers = lsmtree.layers
  plt.axis('equal')
  patches, texts = plt.pie(
    [cache.hits] + [memtbl.hits] + [l.accesses for l in layers],
    labels=['Cache', 'Memtbl'] + ['L{}'.format(i+1) for i in range(len(layers))],
    colors=['green', 'lightgreen'] + [((i+1)*0.2, 0, 0) for i in range(len(layers))])

def plot_access_barchart(lsmtree):
  memtbl = lsmtree.memtbl
  cache = lsmtree.cache
  layers = lsmtree.layers
  plt.bar(np.arange(len(layers)+2)-0.4, [cache.hits,memtbl.hits]+[l.accesses for l in layers])
  plt.xticks(range(len(layers)+2),
             ['Cache', 'Mtbl'] + ['L{}'.format(i+1) for i in range(len(layers))])
  plt.ylabel('# queries that access')
  plt.xlabel('LSM tree component')

def plot_bloom_fp_rates(lsmtree, title=None):
  plt.title('False positive rates for bloom filters of {}'.format(title or lmstree))
  fp_rates = [layer.bloom.false_positive_rate for layer in lsmtree.layers]
  positions = 0.5 - np.arange(len(fp_rates))
  plt.barh(positions, [1]*len(fp_rates), align='center', color='green')
  plt.barh(positions, fp_rates, align='center', color='red')
  plt.yticks(positions, ['L{}'.format(i+1) for i in range(len(fp_rates))])
  plt.xlim((0,1))

def plot_workloads(wls):
  qs = [[q[0] for q in wl.queries] for wl in wls]
  maxy = max(max(q) for q in qs)
  maxx = len(qs[0])
  with figure_grid(1, len(wls)) as grid:
    for q, wl, ax in zip(qs, wls, grid.each_subplot()):
      plt.title(wl)
      plt.xlim(0, maxx)
      if type(wl) == ZipfWorkload:
        plt.yscale("log")
      else:
        plt.ylim(0, maxy)

      qs = list(enumerate(wl.queries))
      read_x = [i for i, x in qs if x[1] == 0]
      read_y = [x[0] for _, x in qs if x[1] == 0]
      write_x = [i for i, x in qs if x[1] == 1]
      write_y = [x[0] for _, x in qs if x[1] == 1]
      plt.scatter(write_x, write_y, alpha=0.025, color="red")
      plt.scatter(read_x, read_y, alpha=0.025, color="blue")
