# CS265 Research Project: Optimizing LSM Tree Memory Allocation

Mali Akmanalp, Sophie Hilgard, and Andrew Ross

## Main Question

For a given workload and initial database state, can we find the optimal allocation of memory to an LSM tree's buffer, cache, and bloom filters?

## Characterizing Workloads

Workloads aren't always reducible to a read-write ratio or a single skewness parameter. We want to characterize workloads as the outcomes of rich stochastic processes or probabilistic models. We want to explore what the "optimal memory allocation" means for different workload models and how it changes as we vary workloads from our model, the model's parameters, and the model itself. 

### Workloads We've Implemented

|Workload name   |Parameters      |
|----------------|----------------|
|Uniform         |`k`, `rw_ratio` |
|Round Robin     |`k`, `rw_ratio` |
|Zipf            |`skewness`, `rw_ratio`|
|Discover-Decay  |`λr`, `λw`, `λu`, `decay_rate`, `popularity`|
|Periodic-Decay  |`λr`, `λw`, `λu`, `decay_rate`, `popularity`, `period`, `cuspiness`|
