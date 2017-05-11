# coding: utf-8
import dill
from matplotlib import pyplot as plt
import glob
from plot_helpers import *

def bloom_variations(prefix):
    return [dill.load(open(prefix+suffix, 'rb'))[3] for suffix in ['-mk.pkl', '-bl.pkl']]

def plot_variations(prefix):
    mk, bl = bloom_variations(prefix)
    compare_cbm_contours(mk, bl, quiver=False)


for filename in glob.glob("./newrun/*-mk.pkl"):

    filename = filename[:-7]
    print(filename)

    plot_variations(filename)
    title = filename.split("/")[-1].replace("\n", " ")
    plt.suptitle(title, y=1.05, fontsize=16)
    plt.show()
