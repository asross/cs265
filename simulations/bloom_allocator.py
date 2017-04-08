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

# class 

# def mk_const_bits_per_layer(m, n):
  # return m, ceil(m*ln2/n)

# def mk_const_bits_per_elem(ratio, n):
  # return const_bits_per_layer_alloc(ratio*n, n)

# def mk_const_false_pos_prob(p, n):
  # lnp = -np.log(p)
  # return ceil(lnp*n/ln22), ceil(lnp/ln2)

"""
function myLog(x, base)
{
    return Math.log(x) / Math.log(base);
}


function calcBits(filter) {
    var denom=Math.pow(Math.log(2),2);
    return filter.fp == 0 ? 0 : ( - filter.nokeys * Math.log(filter.fp) ) / denom;
}


function calcTotalMemBits(filtersArray) {
    var total = 0;
    for (var i = 0; i < filtersArray.length ; i++) {
        // printf("%d   %d  %f \n", i,  rates[i].size, rates[i].false_positive_rate);
        var val = calcBits(filtersArray[i]);

        total += val;
    }
    return total;
}

function getRmax(lt,N,E,B,T)
{
    var R;
    if (lt==0)
        R=Math.ceil(myLog(N*E/B,T));
    else if (lt==1)
        R=(T-1)*Math.ceil(myLog(N*E/B,T));
    return R;
}




function calc_R(f)
{
    var denom = Math.pow(Math.log(2), 2);
    var value = Math.exp(-(f.mem / f.nokeys) * denom);
    return value;
}


// TODO for tiering, we now assume there are (T-1) runs in the last level are all equal to each other in size, 
// but if the last level is not full to capacity, then there may in reality be less than (T-1) runs, and they 
// may have different sizes. To fix this problem, we can insert multiple runs per filter to the filters_array
// until the number of remaining keys is 0. 
function initFilters(N,E,mbuffer,T,mfilter,P,leveltier) {
     mfilter_bits=8*mfilter;

    var filter_array = [];
    var remainingKeys=N;
    var level=0;
    //Calculate the number of keys per level in a almost-full in each level LSM tree
    while (remainingKeys>0)
    {
        level++;
        var levelKeys=Math.ceil(Math.min(Math.pow(T,level)*mbuffer/E,N));
        var newFilter = new Filter();
        newFilter.nokeys=levelKeys;
        newFilter.fp=0.0;
        // console.log("New remaining keys: "+(remainingKeys-levelKeys))
        if (remainingKeys-levelKeys<0)
            newFilter.nokeys=remainingKeys;
        //console.log(newFilter.nokeys)
        filter_array.push(newFilter);
        remainingKeys=remainingKeys-levelKeys;
        // console.log(levelKeys)
    }

    //Initialize the memory per level to be equal
    for (var i=0;i<filter_array.length;i++)
    {
        filter_array[i].mem=mfilter_bits/filter_array.length;
    }   
    return filter_array;
}


function getBaselineFPassigment(N,E,mbuffer,T,mfilter,P,leveltier)
{

    var filter_array = initFilters(N,E,mbuffer,T,mfilter,P,leveltier);

    // console.log(filter_array);
    
    var limit_on_M=mbuffer*8; //amount of memory for filters in bits
    var diff = limit_on_M;
    var change = true;
    var iteration = 0;

    while (diff > 1) {
        change = false;
        for (var i = 0; i < filter_array.length - 1; i++) {
            for (var j = i + 1; j < filter_array.length ; j++) {

                var f1_orig = calc_R(filter_array[i]);
                var f2_orig = calc_R(filter_array[j]);
                // console.log(f1_orig+', '+f2_orig)
                var diff_orig = Math.abs(f1_orig - f2_orig);

                filter_array[i].mem += diff;
                filter_array[j].mem -= diff;

                var f1_new = calc_R(filter_array[i]);
                var f2_new = calc_R(filter_array[j]);
                var diff_new = Math.abs(f1_new - f2_new);
                // console.log(f1_new+', '+f2_new)


                if (diff_new < diff_orig && filter_array[j].mem > 0 && filter_array[i].mem > 0) {
                    change = true;
                    continue;
                }
                filter_array[i].mem -= diff * 2;
                filter_array[j].mem += diff * 2;

                f1_new = calc_R(filter_array[i]);
                f2_new = calc_R(filter_array[j]);
                diff_new = Math.abs(f1_new - f2_new);

                if (diff_new < diff_orig && filter_array[j].mem > 0 && filter_array[i].mem > 0) {
                    change = true;
                    continue;
                }
                filter_array[i].mem += diff;
                filter_array[j].mem -= diff;
            }
        }
        if (!change) {
            diff /= 2;
        }
        iteration++;
    }
    for (var i = 0; i < filter_array.length; i++) {
        filter_array[i].fp = calc_R(filter_array[i]);
        // console.log(filter_array[i].mem+', '+filter_array[i].fp)
    }
    return filter_array;    
}


function getMonkeyFPassigment(N,E,mbuffer,T,mfilter,P,leveltier)
{



    var filter_array = initFilters(N,E,mbuffer,T,mfilter,P,leveltier);

    // console.log(filter_array);
    var limit_on_M=mbuffer*8; //amount of memory for filters in bits
    var diff = limit_on_M;
    var change = true;
    var iteration = 0;
    var current_R = eval_R(filter_array, leveltier, T);
    var original = current_R;
    var value = 0;
    while (diff > 1) {
        change = false;
        for (var i = 0; i < filter_array.length - 1; i++) {
            for (var j = i + 1; j < filter_array.length ; j++) {
                filter_array[i].mem += diff;
                filter_array[j].mem -= diff;
                value = eval_R(filter_array, leveltier, T);
                if (value < current_R && value > 0 && filter_array[j].mem > 0 ) {
                    current_R = value;
                    change = true;
                    continue;
                }
                filter_array[i].mem -= diff * 2;
                filter_array[j].mem += diff * 2;

                value = eval_R(filter_array, leveltier, T);

                if (value < current_R && value > 0 && filter_array[i].mem > 0 ) {
                    current_R = value;
                    change = true;
                    continue;
                }
                filter_array[i].mem += diff;
                filter_array[j].mem -= diff;
            }
        }
        if (!change) {
            diff /= 2;
        }
        iteration++;
    }

    for (var i = 0; i < filter_array.length; i++) {
        filter_array[i].fp = calc_R(filter_array[i]);
    }

    return filter_array;    
}



function eval_R(filters, leveltier, T)
{
	var total = 0; 
    for (var i = 0; i < filters.length ; i++) 
    {
        var val = calc_R(filters[i]);
        if (leveltier == 0) {  // tiering
            total += val * (T-1);
        } else {
            total += val;
        }
    }
    return total;
}
"""
