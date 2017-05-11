import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
from figure_grid import *
from workloads import *
from bloom_assignments import *
from mpl_toolkits.mplot3d import Axes3D
from bloom_assignments import *
from collections import defaultdict

def plot_access_piechart(lsmtree, title=None):
  memtbl = lsmtree.memtbl
  cache = lsmtree.cache
  layers = lsmtree.layers
  plt.axis('equal')
  l = len(layers)
  mems = [cache.hits] + [memtbl.hits] + [l.accesses for l in layers]
  labs = ['Cache', 'Memtbl'] + ['L{}'.format(i+1) for i in range(l)]
  clrs = ['green', 'lightgreen'] + [((i+1)*(1./(l+1)), 0, 0) for i in range(l)]
  patches, texts = plt.pie(mems, colors=colors)
  plt.legend(patches, labels,  loc='left center', bbox_to_anchor=(-0.1, 1.), fontsize=8)

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

def plot_workloads(wls, perrow=4, **kwargs):
  qs = [[q[0] for q in wl.queries] for wl in wls]
  maxys = defaultdict(int)
  for i, q in enumerate(qs):
    maxys[type(wls[i])] = max(maxys[type(wls[i])], max(q))
  maxx = len(qs[0])
  with figure_grid(len(wls)//perrow+len(wls)%perrow, perrow, **kwargs) as grid:
    for q, wl, ax in zip(qs, wls, grid.each_subplot()):
      plt.title(wl.title())
      plt.xlim(0, maxx)
      if type(wl) == ZipfWorkload:
        plt.yscale("log")
      else:
        plt.ylim(0, maxys[type(wl)])
      qs = list(enumerate(wl.queries))
      read_x = [i for i, x in qs if x[1] == 0]
      read_y = [x[0] for _, x in qs if x[1] == 0]
      write_x = [i for i, x in qs if x[1] == 1]
      write_y = [x[0] for _, x in qs if x[1] == 1]
      plt.scatter(write_x, write_y, alpha=0.025, color="green")
      plt.scatter(read_x, read_y, alpha=0.025, color="blue")

def bary_to_cartesian(points):
  points = np.asanyarray(points)
  ndim = points.ndim # will use this to have similar output shape to input
  if ndim == 1: points = points.reshape((1,points.size))
  d = points.sum(axis=1) # in case values aren't normalized
  x = 0.5*(2*points[:,1] + points[:,2])/d
  y = (np.sqrt(3.0)/2) * points[:,2]/d
  out = np.vstack([x,y]).T
  if ndim == 1: return out.reshape((2,))
  return out

def cbm_results(trees):
  return np.array([(t.memtbl.size, t.cache.size, np.log10(t.disk_accesses)) for t in trees]).T

def plot_cbm_trisurf(trees,**kwargs):
  X,Y,Z = cbm_results(trees)
  i = Z.argmin()
  plt.gca().plot_trisurf(X,Y,Z,alpha=0.25,**kwargs)
  plt.scatter([X[i]],[Y[i]], zs=[Z[i]], s=50, c=kwargs.get('color', 'yellow'))
  plt.xlabel('Memtbl')
  plt.ylabel('Cache')
  plt.gca().set_zlabel('Disk')

def plot_cbm_contourf(trees,**kwargs):
  X,Y,Z = cbm_results(trees)
  i = Z.argmin()
  plt.tricontourf(tri.Triangulation(X,Y), Z, 100, **kwargs)
  plt.scatter(X[i],Y[i], s=50, c=kwargs.get('color', 'yellow'))
  plt.xlabel('Memtbl')
  plt.ylabel('Cache')

rt32 = np.sqrt(3)/2.

def savings_triples(trees, **kwargs):
  return np.array([(t.bigger_cache_savings(p=0.5), t.bigger_memtbl_savings(), t.bigger_bloom_savings(**kwargs)) for t in trees])

def savings_pairs(trees, **kwargs):
  preds = savings_triples(trees, **kwargs)
  maxes = preds.argmax(axis=1)
  minis = preds.argmin(axis=1)
  order = ['cache', 'memtbl', 'bloom']
  return [(order[sm], order[lg]) for sm, lg in zip(minis, maxes)]

def arrows_for(savepairs):
  redist = { ('memtbl', 'cache'): np.array([1,0]),
             ('cache', 'bloom'): np.array([-.5,.5*np.sqrt(3)]),
             ('bloom', 'memtbl'): np.array([-.5,-.5*np.sqrt(3)])}
  def arrow_for(p): return redist[p] if p in redist else -redist[p[::-1]]
  return np.array([arrow_for(p) for p in savepairs])

def plot_cbm_simplex(trees,ballocs=monkey_assignment,dM=100,quiver=True,paths=False,**kwargs):
  X,Y,Z = cbm_results(trees)
  M = min(X)+max(Y)
  i = Z.argmin()
  plt.axis('equal')
  plt.axis('off')

  # convert to simplex
  C = bary_to_cartesian(np.vstack((X/M, Y/M, 1-X/M-Y/M)).T)

  # contour plot of experimental results
  plt.tricontourf(tri.Triangulation(C[:,0], C[:,1]), Z, 100, **kwargs)

  # quiver plot of estimated gradients
  if quiver:
    arrows = arrows_for(savings_pairs(trees, ballocs=ballocs))
    plt.quiver(C[:,0], C[:,1], arrows[:,0], arrows[:,1], color='black', alpha=0.5)

  if paths:
    paths = savings_paths(trees, dM=dM, ballocs=ballocs)
    for path in paths:
      plt.plot(C[path,0], C[path,1], color='yellow', alpha=0.25, lw=2)

  # experimental minimum
  plt.scatter(C[i,0], C[i,1], s=50, c=kwargs.get('color', 'yellow'))

  # dashed outline
  corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
  triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
  plt.triplot(triangle, linewidth=1, linestyle='--', color='black', alpha=0.1)

  # labels
  plt.text(*[-0.05, 0], 'Buffer', {'ha': 'center', 'va': 'center'}, rotation=-60)
  plt.text(*[1.05, 0], 'Cache', {'ha': 'center', 'va': 'center'}, rotation=60)
  plt.text(*corners[2], 'Bloom', {'ha': 'center', 'va': 'center'})

def compare_cbm_trisurfs(monkey, baseline, ballocs=None, ax=None):
  if ax is None:
    ax = plt.subplot(111, projection='3d')
  plot_cbm_trisurf(monkey, color='blue')
  plot_cbm_trisurf(baseline, color='red')

def compare_cbm_contours(monkey, baseline, quiver=True, paths=False, dM=100, figsize=(10,4)):
  _x1,_y1, Z1 = cbm_results(monkey)
  _x2,_y2, Z2 = cbm_results(baseline)
  Zmin = min(Z1.min(), Z2.min())
  Zmax = max(Z1.max(), Z2.max())
  norm = matplotlib.colors.Normalize(Zmin, Zmax)

  fig = plt.figure(figsize=figsize)
  ax1 = plt.subplot(121)
  plt.title('Monkey', y=1.05)
  plot_cbm_simplex(monkey, norm=norm, ballocs=monkey_assignment, quiver=quiver, paths=paths, dM=dM)
  ax2 = plt.subplot(122)
  plt.title('Baseline', y=1.05)
  plot_cbm_simplex(baseline, norm=norm, ballocs=baseline_assignment, quiver=quiver, paths=paths, dM=dM)

  cbaxes = fig.add_axes([0.5, 0.1, 0.03, 0.8])
  cbaxes.set_title(r'$\log_{10}(Disk)$')
  m = cm.ScalarMappable()
  m.set_array(np.hstack((Z1,Z2)))
  cb = plt.colorbar(m, cax = cbaxes)

def savings_paths(trees, dM, **kwargs):
  results = cbm_results(trees).T
  savepairs = savings_pairs(trees, **kwargs)
  redist = { ('memtbl', 'cache'): np.array([-dM, dM]),
             ('cache', 'bloom'): np.array([0, -dM]),
             ('bloom', 'memtbl'): np.array([dM, 0]) }
  def redist_for(p):
    return redist[p] if p in redist else -redist[p[::-1]]
  def next_i(i):
    table1, cache1, _ = results[i]
    dt, dc = redist_for(savepairs[i])
    table2 = table1 + dt
    cache2 = cache1 + dc
    for j, (table, cache, _) in enumerate(results):
      if table == table2 and cache == cache2:
        return j
  paths = []
  for i in range(len(trees)):
    path = [i]
    visited = set([i])
    while True:
      j = next_i(i)
      if j is None or j in visited:
        break
      else:
        path.append(j)
        visited.add(j)
        i = j
    paths.append(path)
  return paths
