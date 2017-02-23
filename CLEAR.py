'''
Copyleft Feb 22, 2017 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
import os,sys,optparse
import numpy as np;
import pandas as pd;
import seaborn as sns
import pylab as plt;
import matplotlib as mpl

sys.path.insert(1,os.getcwd())
np.set_printoptions(linewidth=200, precision=5, suppress=True)
pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False


parser = optparse.OptionParser()
parser.add_option( '--sync', action="store", dest="syncFile", help="path to synchronized file created by popoolation2")
parser.add_option( '--pandas', action="store", dest="pandasFile", help="path to pandas dataframe")
options, args = parser.parse_args()


if __name__ == '__main__':
    print options
