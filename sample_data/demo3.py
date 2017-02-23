'''
Copyleft Feb 22, 2017 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
import numpy as np;

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import seaborn as sns
import pylab as plt;
import matplotlib as mpl
import os;

home = os.path.expanduser('~') + '/'
import popgen.Util as utl
import popgen.Estimate as est
import popgen.Simulation as Simulation