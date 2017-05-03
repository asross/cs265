"""
To run this:
    - pip install bokeh
    - bokeh serve visualization.py
    - go to http://localhost:5006/visualization
"""


from lsmulator import LSMulator
from bloom_assignments import monkey_assignment
from workloads import ZipfWorkload

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox, column, layout
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, Button, Toggle
from bokeh.plotting import figure

# Data sources
workload_reads = ColumnDataSource(data=dict(x=[], y=[]))
workload_writes = ColumnDataSource(data=dict(x=[], y=[]))
lsm_components = ColumnDataSource(data=dict(queries=[], layer=[]))


# Set up plots
workload_plot = figure(title="Workload Characteristics",
                       tools="crosshair,pan,reset,save,wheel_zoom", y_axis_type="log"
                       )

workload_plot.xaxis.axis_label = "Time"
workload_plot.yaxis.axis_label = "Key"

workload_plot.circle('x', 'y', source=workload_writes, legend="Writes", color="red")
workload_plot.circle('x', 'y', source=workload_reads, legend="Reads")

layer_plot = figure(title="Hits per level of tree",
                    tools="crosshair,pan,reset,save,wheel_zoom",
                    x_range=["Cache", "Mtbl", "L1"],
                    y_range=[0, 500],
                    )

layer_plot.xaxis.axis_label = "LSM Tree Component"
layer_plot.yaxis.axis_label = "# queries that access"

layer_plot.segment('layer', 0, 'layer', 'queries', line_width=40, source=lsm_components)


# Set up widgets
zipf_ratio = Slider(title="Zipf Ratio", value=1.2, start=1.01, end=2.5, step=0.05)

layer_ratio = Slider(title="Layer Ratio", value=2.0, start=2.0, end=8.0, step=0.5)
total = Slider(title="Total Size", value=200.0, start=100.0, end=1000.0)
memtbl_size = Slider(title="Memtable Size", value=100.0, start=50.0, end=1000.0)
bloom = Slider(title="Bloom Size", value=50, start=10, end=1000.0)

refresh_button = Button(label="Refresh")

def update_graphs(attrname, old, new):

    # Generate workload
    workload = ZipfWorkload(n_queries=1000, zipf_param=zipf_ratio.value)

    # Precalculate number of layers
    num_layers = LSMulator.emulate(workload.queries, memtbl_size=memtbl_size.value,
                                   layer_ratio=layer_ratio.value).layer_sizes

    # Create tree
    tree = LSMulator.emulate(workload.queries,
                             layer_ratio=layer_ratio.value,
                             memtbl_size=memtbl_size.value,
                             cache_size=total.value - memtbl_size.value - bloom.value,
                             bloom_size=monkey_assignment(bloom.value, memtbl_size.value, num_layers))

    # Update workload graph
    qs = list(enumerate(workload.queries))
    reads = [(i, x[0]) for i, x in qs if x[1] == 0]
    writes = [(i, x[0]) for i, x in qs if x[1] == 1]
    workload_reads.data = dict(y=[x for i, x in reads], x=[i for i, x in reads])
    workload_writes.data = dict(y=[x for i, x in writes], x=[i for i, x in writes])

    # Update LSM components graph
    accesses = [tree.cache.hits, tree.memtbl.hits] + [l.accesses for l in tree.layers]
    layers = ['Cache', 'Mtbl'] + ['L{}'.format(i+1) for i in range(len(tree.layers))]
    print(layers, accesses)
    print([(x.title, x.value) for x in controls])

    layer_plot.x_range.factors = layers
    lsm_components.data = dict(layer=layers, queries=accesses)


controls = [zipf_ratio, layer_ratio, memtbl_size, total, bloom]

# Set up when the graph should redraw
#refresh_button.on_click(lambda: update_workload(None, None, None))
[c.on_change("value", update_graphs) for c in controls]

# Set up layouts and add to document
workload_inputs = widgetbox(zipf_ratio)
memtable_inputs = widgetbox(layer_ratio, total, memtbl_size, bloom)

curdoc().add_root(layout(children=[
    [workload_inputs, memtable_inputs, refresh_button],
    [workload_plot, layer_plot]
]
))
curdoc().title = "Sliders"

# Load for the first time
update_graphs(None, None, None)
