from glob import glob
import numpy as np
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt

proc_dps = glob('testing_wpipe/emcee/data/testdata/proc_conf/M_*.csv')
EXISTING_MODELS = pd.concat([pd.read_csv(proc_dp).set_index(['a0', 'a1']) for proc_dp in proc_dps])
CHARA_LENGTHS = 1.
args = (0., 0.)
models = EXISTING_MODELS.drop('name', axis=1)
deviations = (np.array(models.index.to_list()) - args) / CHARA_LENGTHS
vor = spatial.Voronoi(deviations, incremental=True)
fig = spatial.voronoi_plot_2d(vor)
fig.show()
plt.pause(.001)
