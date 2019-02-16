import numpy as np
import pandas as pd

from UTILS import *
from VCF import gz,VCF


def loadPiarPop(f,pop,popxp,negate=False):
    load=pd.read_pickle
    if f[-3:]=='.gz':load=gz.load
    try:return load(f.format(pop, popxp))
    except:
        alpha=(1,-1)[negate]
        return load(f.format(popxp, pop))*alpha
class GENOME:
    @staticmethod
    def after(x, pos=1e7):
        if len(x.index.names)==1:
            x.index.name='POS'
        if pos > 0:
            return x[x.index.get_level_values('POS') > pos]
        else:
            return x[x.index.get_level_values('POS') < abs(pos)]

    def __init__(self,assembly=38,dmel=False,faPath='{}storage/Data/Human/ref/'.format(home)):
        "conda install -c bioconda pysam "
        import pysam
        self.assembly=assembly
        organism=('hg','dmel')[dmel]
        self.name='{}{}'.format(organism,self.assembly)
        GENOMEFA = '{}{}.fa'.format(faPath,self.name)
        self.g = pysam.Fastafile(GENOMEFA)
    def chrom(self, a,CHROM):
        return pd.DataFrame(a.groupby(level=0).apply(lambda x: self.base(CHROM,x.name)))#.loc[CHROM]#.rename(self.name)

    def base(self,CHROM,POS):
        return self.g.fetch('chr{}'.format(CHROM), POS - 1, POS).upper()
    def genome(self,a,join=True):
        b=a.groupby(level=0).apply(lambda x: self.chrom(a.loc[x.name], x.name))[0].rename(self.name)
        if join:
            b=GENOME.join(pd.DataFrame(b),a,CHROMS=a.index.get_level_values(0).unique().tolist())
        b.index.names = ["CHROM", "POS"]
        return b

    @staticmethod
    def mergeCHROM(a, verbose=False, keys=None):
        """
        :param a: list of series each of which is a chromosome
        :return:
        """
        a = [x for x in a if x is not None]
        if not len(a): return None
        CHROM = a[0].index[0][0]
        if verbose: print(CHROM)
        b = pd.concat([pd.concat([dedup(x.loc[CHROM]) for x in a], 1, keys=keys)], keys=[CHROM])
        b.index.names = ['CHROM', 'POS']
        return b

    @staticmethod
    def merge(a, CHROMS=range(1, 23), keys=None):
        if CHROMS is None: CHROMS = a[0].index.levels[0]
        def xs(x, c):
            try:
                return x.loc[[c]]
            except:
                pass

        a = [GENOME.mergeCHROM([xs(x, c) for x in a], keys=keys) for c in CHROMS]
        a = [x for x in a if x is not None]
        return pd.concat(a)

    @staticmethod
    def joinCHROM(a, b, how, verbose=False):
        """
        :param a: list of series each of which is a chromosome
        :return:
        """
        if a is None: return
        if a.shape[0] == 0: return
        CHROM = a.index[0][0]
        return pd.concat([a.loc[CHROM].join(b.loc[CHROM], how=how)], keys=[CHROM])

    @staticmethod
    def join(a, b, CHROMS=range(1, 23), how='inner'):
        if CHROMS is None:  CHROMS = a.index.get_level_values(0).unique().tolist()
        if how == 'inner':    CHROMS = b.index.get_level_values(0).unique().tolist()

        def xs(x, c):
            try:
                return x.loc[[c]]
            except:
                pass

        a = [GENOME.joinCHROM(xs(a, c), xs(b, c), how=how) for c in CHROMS]
        a = [x for x in a if x is not None]
        a = pd.concat(a)
        a.index.names = ['CHROM', 'POS']
        return a

    @staticmethod
    def safeConcat(a, keys=None):
        return pd.concat([x for x in a if x is not None], keys=keys)

    @staticmethod
    def filterGapChr(a, chr, GAP):
        b = a.loc[chr]
        gap = GAP.loc[chr]
        gap['len'] = gap.end - gap.start
        return b.drop(pd.concat([maskChr(b, i) for _, i in gap.iterrows()]).index)

    @staticmethod
    def filterGapChr(a, CHROM, gap):
        b = a.loc[CHROM]
        return b.drop(pd.concat([maskChr(b, i) for _, i in gap.loc[CHROM].iterrows()]).index)

    @staticmethod
    def filterGap(a, assempbly=19, pad=200000):
        CHROMS = a.index.get_level_values('CHROM').unique()
        gap = loadGap(assempbly, pad)
        return pd.concat([GENOME.filterGapChr(a, chr, gap) for chr in CHROMS], keys=CHROMS)

class scan:
    @staticmethod
    def cdf(x):
        import pylab as plt
        ax=plt.subplots(1,2,figsize=(8,3),dpi=1)[1]
        # sns.distplot(x,ax=ax[0])
        CDF(x).plot(label='CDF',lw=4,c='k',alpha=0.75, ax=ax[1]);
        c='darkblue'
        ax[1].axvline(x.quantile(0.5),c=c,alpha=0.5,label='Median={}'.format(x.quantile(0.5)));
        ax[1].axvline(x.quantile(0.95),c=c,ls='--',alpha=0.5,label='Q95     ={}'.format(x.quantile(0.95)));
        ax[1].axvline(x.quantile(0.99), c=c,ls='-.', alpha=0.5, label='Q99     ={}'.format(x.quantile(0.99)));
        ax[1].legend();
    @staticmethod
    def topK(x, k=2000):
        return x.sort_values(ascending=False).iloc[:k]
    @staticmethod
    def idf(a, winSize=50000, names=None):
        if names == None: names = [a.name, 'n']
        x=scan.Genome(a.dropna(), f={names[0]: np.mean, names[1]: len}, winSize=winSize)
        x.columns=[0,'n']
        return x

    @staticmethod
    def Genome(genome, f=lambda x: x.mean(), uf=None,winSize=50000, step=None, nsteps=5, minSize=None):
        """
        Args:
            genome: scans genome, a series which CHROM and POS are its indices
            windowSize:
            step:
            f: is a SCALAR function or dict of SCALAR fucntions e.g. f= {'Mean' : np.mean, 'Max' : np.max, 'Custom' : np.min}
            Only good for scanning a series with dictionary of scalar fucntions
            uf: is a universal function which returns a dataframe e.g. uf=lambda x: pd.DataFrame(np.random.rand(2,3))
            good for scanning a dataframe (which each column to be scanned) with a scalar or  univesal fucntions
        Returns:
        """
        if len(genome.shape)>1:
            return genome.apply(lambda x: scan.Genome(x,f=f,uf=uf,winSize=winSize,step=step,nsteps=nsteps))

        if step is None:step=winSize/nsteps
        df = genome.groupby(level='CHROM').apply(lambda ch: scan.Chromosome(ch.loc[ch.name],f,uf,winSize,step))
        if minSize is not None:
            n=scan.Genome(genome, f=lambda x: x.size, winSize=winSize, step=step, minSize=None)
            if f==np.sum:
                df=df.loc[TI(n>=minSize)]
            else:
                df=df[n>=minSize]
        return df

    @staticmethod
    def Chromosome(x,f=np.mean,uf=None,winSize=50000,step=10000):
        """
        Args:
            chrom: dataframe containing chromosome, positions are index and the index name should be set
            windowSize: winsize
            step: steps in sliding widnow
            f: is a SCALAR function or dict of SCALAR fucntions e.g. f= {'Mean' : np.mean, 'Max' : np.max, 'Custom' : np.min}
            uf: is a universal function which returns a dataframe e.g. uf=lambda x: pd.DataFrame(np.random.rand(2,3))
        Returns:
        """
        # print 'Chromosome',x.name
        if x.index[-1] - x.index[0] < winSize:
            f=(f,uf)[uf is not None]
            i= roundto(((x.index[-1] + x.index[0]) / 2.),10000)+5000
            z=pd.DataFrame([f(x)], index=[i])
            z.index.name='POS'
            return z

        POS=x.index.get_level_values('POS')
        res=[]
        # Bins=np.arange(max(0,roundto(POS.min()-winSize,base=step)), roundto(POS.max(),base=step),winSize)
        Bins = np.arange(0, roundto(POS.max(), base=step), winSize)
        for i in range(int(winSize/step)):
            bins=i*step +Bins
            windows=pd.cut( POS, bins,labels=(bins[:-1] + winSize/2).astype(int))
            if uf is None:
                tmp=x.groupby(windows).agg(f)
                tmp.index=tmp.index.astype(int);
                tmp.index.name='POS'

            else:
                tmp=x.groupby(windows).apply(uf)
                tmp=tmp.reset_index()
                tmp.iloc[:,0]=tmp.iloc[:,0].astype(int)
                tmp.columns=['POS']+tmp.columns[1:].tolist()
                tmp= tmp.set_index(tmp.columns[:-1].tolist()).iloc[:,0]
            res+=[tmp]
        df=pd.concat(res).sort_index().dropna()
        # if minSize is not None:
        #     df[df.COUNT < minSize] = None
        #     df = df.loc[:, df.columns != 'COUNT'].dropna()
        return df

    @staticmethod
    def scanGenomeSNP(genome, f=np.mean, winSize=300,skipFromFirst=0,step=None):
        if step is None:step=int(winSize/5)
        return  genome.groupby(level=0).apply(lambda x: scan.ChromosomeSNP(x.iloc[skipFromFirst:],f,winSize,step))

    @staticmethod
    def scanChromosomeSNP(x,f,winSize,step):
        """
        Args:
            chrom: dataframe containing chromosome, positions are index and the index name should be set
            windowSize: winsize
            step: steps in sliding widnow
            f: is a function or dict of fucntions e.g. f= {'Mean' : np.mean, 'Max' : np.max, 'Custom' : np.min}
        Returns:
        """
        BinsStart=pd.Series(np.arange(0, roundto(x.size,base=step),winSize),name='start')
        def createBins(i):
            bins=pd.DataFrame(i*step +BinsStart)
            bins['end'] = bins.start+ winSize
            bins.index=((bins.start+bins.end)/2).astype(int)
            return bins
        bins=pd.concat(map(createBins,range(int(winSize/step)))).sort_index()
        bins[bins>x.size]=None
        bins=bins.dropna().astype(int)
        bins=bins.apply(lambda bin: f(x.iloc[range(bin.start,bin.end)]),axis=1)
        bins.index=x.index[bins.index]
        if bins.shape[0]:return bins.loc[x.name]

    @staticmethod
    def smooth(a, winsize, normalize=True):
        if normalize:
            f = lambda x: x / x.sum()
        else:
            f = lambda x: x
        return scan.scan3way(f(a), winsize, np.mean)

    @staticmethod
    def threeWay(a, winsize, f):
        return pd.concat([a.rolling(window=winsize).apply(f),
                          a.rolling(window=winsize, center=True).apply(f),
                          a.iloc[::-1].rolling(window=winsize).apply(f).iloc[::-1]],
                         axis=1)

    @staticmethod
    def scan3way(a, winsize, f):
        return scan.threeWay(a, winsize, f).apply(lambda x: np.mean(x), axis=1)

    @staticmethod
    def scan2wayLeft(a, winsize, f):
        """Moving average with left ellements and centered"""
        X = scan.threeWay(a, winsize, f)
        x = X[[0, 1]].mean(1)
        x[x.isnull] = x[2]
        return x

    @staticmethod
    def scan2wayRight(a, winsize, f):
        """Moving average with left ellements and centered"""
        return scan.threeWay(a, winsize, f).iloc[:, 1:].apply(lambda x: np.mean(x), axis=1)

    @staticmethod
    def plotBestFly(windowStat, X,  pad=30000, i=None, mann=True, foldOn=None,rep=None):
        # i0 = (x.sum(1) > 0.05) & (x.sum(1) < 6.95)
        if rep is None: x=X
        else: x=X.xs(rep,1,1)
        if i is None:
            i = BED.intervali(windowStat.dropna().sort_values().index[-1], pad);
        import UTILS.Plots as pplt
        pplt.Trajectory.Fly(mask(x, i), subsample=2000, reps=[1, 2, 3], foldOn=foldOn);
        # plt.title('Rep {}, {} '.format(rep, utl.BED.strMbp(i)));plt.show()
        if mann: pplt.Manhattan(windowStat, top_k=1)
        return BED.str(i)


def scanXPSFS(pops=['CEU','CHB'],nProc=8):
    from itertools import product
    from multiprocessing import Pool
    try:
        exit()
        return loadPiarPop(PATH.scan + 'SFS/{}.{}.df',pops[0], pops[1])
    except:
        fname = PATH.scan + 'SFS/{}.{}.df'.format(pops[0], pops[1])
        CHROMS=range(1,23)
        pool = Pool(nProc)
        a=pd.concat(pool.map(scanXPSFSChr,product([pops],CHROMS))).sort_index()
        pool.terminate()
        a.to_pickle(fname)
        return a

def scanXPSFSChr(args):
    pops, CHROM=args
    import UTILS.Estimate as est

    df = gz.loadFreqChrom(pops, str(CHROM))
    N=pd.concat(map(lambda x: pd.Series({x:len(VCF.ID(x))}),pops))*2
    w=N/N.sum()
    df=df.join(df.dot(w).rename('all'))
    N['all']=N.sum()
    N = (1 / df[df > 0].min()).astype(int)
    removeFixedSites = False;
    winSize = 5e4
    f = lambda x: pd.DataFrame(scan.Genome(x[x.name],
                                              uf=lambda X: est.Estimate.getEstimate(X.dropna(), n=N[x.name], bins=20,
                                                                                    removeFixedSites=removeFixedSites,
                                                                                    normalizeTajimaD=False),
                                              winSize=int(winSize)))
    a=df.groupby(level=0, axis=1).apply(f).T.reset_index(level=0, drop=True).T
    n = df[(df > 0) & (df < 1)].apply(lambda x: scan.Genome(x.dropna(), len))
    n['stat'] = 'n'
    a = pd.concat([n.set_index('stat', append=True), a]).sort_index()
    return a

