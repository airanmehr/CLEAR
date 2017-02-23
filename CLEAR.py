'''
Copyleft Feb 22, 2017 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
import os,sys,optparse
import numpy as np;
import pandas as pd;
import seaborn as sns
import pylab as plt;
import matplotlib as mpl
import Utils.Util as utl
import Utils.Plots as pplt
import Libs.Markov as mkv
sys.path.insert(1,os.getcwd())
np.set_printoptions(linewidth=200, precision=5, suppress=True)
pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False


parser = optparse.OptionParser()
parser.add_option( '--sync', action="store", dest="syncFile", help="path to synchronized file created by popoolation2")
parser.add_option( '--pandas', action="store", dest="pandasFile", help="path to pandas dataframe")
options, args = parser.parse_args()


if __name__ == '__main__':
    if options.pandasFile is not None:
        CD=pd.read_pickle(options.pandasFile)
        fout=options.pandasFile+'.out.df'
    elif options.syncFile is not None:
        CD=utl.SynchronizedFile.load(options.syncFile)
        fout=options.syncFile+'.out.df'
    else:
        print 'Invalid input'
        exit()

    HMM=mkv.HMM(CD=CD,gridH=[0.5],saveCDE=False,loadCDE=False,verbose=-1)
    rangeN=10**np.arange(2,7)
    print 'Performing coarse grid search on N=',rangeN
    a=HMM.fitN(rangeN=rangeN,n=200).sort_index(1).mean(0)

    rangeN=sorted(a.sort_values().index[::-1][:2])
    a=a.reset_index();a.columns=['N','Likelihood'];print a
    rangeN=map(lambda x: utl.roundto(x,50),np.linspace(rangeN[0],rangeN[1],10))
    print 'Performing fine grid search on N=',rangeN
    a=HMM.fitN(rangeN=rangeN,n=200).sort_index(1).mean(0)
    N=a.idxmax()
    a=a.reset_index();a.columns=['N','Likelihood'];print a
    print  'Maximum Likelihood of N=',N
    HMM=mkv.HMM(CD=CD,gridH=[0.5],N=N,n=200,saveCDE=False,loadCDE=False,verbose=1)
    a= HMM.fit(False)
    print a
    a.to_pickle(fout)
    print 'Output is saved in pandas dataframe in {}.'.format(fout)

    f=lambda x: x.alt-x.null
    a=f(pd.read_pickle(fout)[0.5])
    fig,axes=plt.subplots(2,1,sharex=True,dpi=200)
    pplt.Manhattan(a.rename('$H$'),top_k=10,axes=[axes[0]])
    pplt.Manhattan(utl.scanGenome(a).rename(r'$\mathcal{H}$'),top_k=3,axes=[axes[1]])
    plt.show()
