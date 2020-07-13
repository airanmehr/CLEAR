import numpy as np
import pandas as pd
import numba
import pylab as plt
import matplotlib as mpl
import os
def mkdir(path):os.system('mkdir -p {}'.format(path))
def roundto(x, base=50000):
    return int(base * np.round(float(x)/base))

def TI(a):
    return a.replace({False:None}).dropna().index

def polymorphic(data, minAF=1e-9,mincoverage=10,index=True):
    def poly(x):return (x>=minAF)&(x<=1-minAF)
    C,D=data.xs('C',level='READ',axis=1),data.xs('D',level='READ',axis=1)
    I=(C.sum(1)/D.sum(1)).apply(lambda x:poly(x)) & ((D>=mincoverage).mean(1)==1)
    if index:
        return I
    return data[I]
def batch(iterable, n=10000000):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
@numba.vectorize
def vectorizedLog(x):
    return float(np.log(x))
def numbaLog(df):
    return  pd.DataFrame(vectorizedLog(df.values),columns=df.index,index=df.index).astype(float)

@numba.vectorize
def vectorizedExp(x):
    return float(np.exp(x))
def numbaExp(df):
    return  pd.DataFrame(vectorizedExp(df.values),columns=df.index,index=df.index).astype(float)

class EE:
    @staticmethod
    def fx(x, s=0.0, h=0.5):
        Z=(1 + s) * x ** 2 + 2 * (1 + h * s) * x * (1 - x) + (1 - x) ** 2
        if Z>0:
            return ((1 + s) * x ** 2 + (1 + h * s) * x * (1 - x)) / (Z)
        else:
            return 0

    @staticmethod
    def sig(x): return 1. / (1 + np.exp(-x))

    @staticmethod
    def logit(p): return np.log(p) - np.log(1 - p)


    # def logit_(p): return T.log(p) - T.log(1 - p)


    # def sig_(x): return 1. / (1 + T.exp(-x))

    @staticmethod
    def Nu(s, t, nu0, theta, n=2000): return EE.Z(EE.sig(t * s / 2 + EE.logit(nu0)), n, theta)

    @staticmethod
    def forward(x0=0.005,h=0.5,s=1,t=150):
        def f(x,h=0.5,s=1): return ((1+s)*x*x + (1+h*s)*x*(1-x) )/((1+s)*x*x + 2*(1+h*s)*x*(1-x)  +(1-x)**2)
        x=[x0]
        for i in range(t):
            x+=[f(x[-1],h,s)]
        return pd.Series(x)

    floatX = 'float64'

    @staticmethod
    def Z(nu, n, theta): return theta * (
    nu * ((nu + 1) / 2. - 1. / ((1 - nu) * n + 1)) + (1 - nu) * ((n + 1.) / (2 * n) - 1. / ((1 - nu) * n + 1)))

class VCF:
    @staticmethod
    def loadDP(fname):
        a= pd.read_csv(fname,sep='\t',na_values='.').set_index(['CHROM','POS'])
        a.columns=pd.MultiIndex.from_tuples(map(lambda x:(int(x.split('R')[1].split('F')[0]),int(x.split('F')[1])),a.columns))
        return  a

    @staticmethod
    def loadCD(vcfgz,vcftools='~/bin/vcftools_0.1.13/bin/vcftools'):
        """
            vcfgz: vcf file where samples are in the format of RXXFXXX
        """
        vcf=os.path.basename(vcfgz)
        path=vcfgz.split(vcf)[0]
        os.system('cd {0} && {1} --gzvcf {2} --extract-FORMAT-info DP && {1} --gzvcf {2} --extract-FORMAT-info AD'.format(path,vcftools,vcf))
        fname='out.{}.FORMAT'
        a=map(lambda x: VCF.loadDP(path +fname.format(x)) ,['AD','DP'])
        a=pd.concat(a,keys=['C','D'],axis=1).reorder_levels([1,2,0],1).sort_index(1)
        a.columns.names=['REP','GEN','READ']
        return a


class SynchronizedFile:
    @staticmethod
    def processSyncFileLine(x,dialellic=True):
        z = x.apply(lambda xx: pd.Series(xx.split(':'), index=['A', 'T', 'C', 'G', 'N', 'del'])).astype(float).iloc[:, :4]
        ref = x.name[-1]
        alt = z.sum().sort_values()[-2:]
        alt = alt[(alt.index != ref)].index[0]
        if dialellic:   ## Alternate allele is everthing except reference
            return pd.concat([z[ref].astype(int).rename('C'), (z.sum(1)).rename('D')], axis=1).stack()
        else:           ## Alternate allele is the allele with the most reads
            return pd.concat([z[ref].astype(int).rename('C'), (z[ref] + z[alt]).rename('D')], axis=1).stack()

    @staticmethod
    def load(fname = './sample_data/popoolation2/F37.sync'):
        # print 'loading',fname
        cols=pd.read_csv(fname+'.pops', sep='\t', header=None, comment='#').iloc[0].apply(lambda x: map(int,x.split(','))).tolist()
        data=pd.read_csv(fname, sep='\t', header=None).set_index(range(3))
        data.columns=pd.MultiIndex.from_tuples(cols)
        data.index.names= ['CHROM', 'POS', 'REF']
        data=data.sort_index().reorder_levels([1,0],axis=1).sort_index(axis=1)
        data=data.apply(SynchronizedFile.processSyncFileLine,axis=1)
        data.columns.names=['REP','GEN','READ']
        data=SynchronizedFile.changeCtoAlternateAndDampZeroReads(data)
        data.index=data.index.droplevel('REF')
        return data

    @staticmethod
    def changeCtoAlternateAndDampZeroReads(a):
        C = a.xs('C', level=2, axis=1).sort_index().sort_index(axis=1)
        D = a.xs('D', level=2, axis=1).sort_index().sort_index(axis=1)
        C = D - C
        if (D == 0).sum().sum():
            C[D == 0] += 1
            D[D == 0] += 2
        C.columns = pd.MultiIndex.from_tuples([x + ('C',) for x in C.columns], names=C.columns.names + ['READ'])
        D.columns = pd.MultiIndex.from_tuples([x + ('D',) for x in D.columns], names=D.columns.names + ['READ'])
        return pd.concat([C, D], axis=1).sort_index(axis=1).sort_index()




def Manhattan(data, columns=None, names=None, fname=None, colors=['black', 'gray'], markerSize=20, ylim=None, show=True,
              std_th=None, top_k=None, cutoff=None, common=None, Outliers=None, shade=None, fig=None, ticksize=16,
              sortedAlready=False,lw=1,axes=None,shareY=False,color=None,CHROMLen=None):
    def reset_index(x):
        if x is None: return None
        if 'CHROM' not in x.columns.values:
            return x.reset_index()
        else:
            return x
    if type(data) == pd.Series:
        DF = pd.DataFrame(data)
    else:
        DF = data

    if columns is None: columns=DF.columns
    if names is None:names=columns

    df = reset_index(DF)
    Outliers = reset_index(Outliers)
    if not sortedAlready: df = df.sort_index()
    if not show:
        plt.ioff()
    from itertools import cycle
    def plotOne(b, d, name, chroms,common,shade,ax):
        a = b.dropna()
        c = d.loc[a.index]
        if ax is None:
            ax=plt.gca()
        if shade is not None:
            for _ ,  row in shade.iterrows():
                if shareY:
                    MAX = DF.replace({np.inf: None}).max().max()
                    MIN = DF.replace({-np.inf: None}).min().min()
                else:
                    MAX = a.replace({np.inf: None}).max()
                    MIN = a.replace({-np.inf: None}).min()
                ax.fill_between([row.gstart, row.gend], MIN,MAX, color='b', alpha=0.4)

                if 'name' in row.index:
                    if row['name'] == 1: row.gstart -=  1e6
                    if row['name']== 8: row.gstart=row.gend+1e6
                    xy=(row.gstart, (MAX*1.1))
                    try:shadename=row['name']
                    except:shadename=row['gene']
                    ax.text(xy[0],xy[1],shadename,fontsize=ticksize+2,rotation=0,ha= 'center', va= 'bottom')
                    # ax.annotate('   '+shadename,
                    #             # bbox=dict(boxstyle='round,pad=1.2', fc='yellow', alpha=0.3),
                    #             xy=xy, xytext=xy, xycoords='data',horizontalalignment='center',fontsize=ticksize,rotation=90,verticalalignment='bottom')

        ax.scatter(a.index, a, s=markerSize, c=c, alpha=0.8, edgecolors='none')

        outliers=None
        if Outliers is not None:
            outliers=Outliers[name].dropna()
        if cutoff is not None:
            outliers = a[a >= cutoff[name]]
        elif top_k is not None:
            outliers = a.sort_values(ascending=False).iloc[:top_k]
        elif std_th is not None:
            outliers = a[a > a.mean() + std_th * a.std()]
        if outliers is not None:
            if len(outliers):
                ax.scatter(outliers.index, outliers, s=markerSize, c='r', alpha=0.8, edgecolors='none')
                # ax.axhline(outliers.min(), color='k', ls='--',lw=lw)


        if common is not None:
            for ii in common.index: plt.axvline(ii,c='g',alpha=0.5)

        ax.axis('tight');
        if CHROMLen is not None:
            ax.set_xlim(0, CHROMLen.sum());
        else:
            ax.set_xlim(max(0,a.index[0]-10000), a.index[-1]);
        setSize(ax,ticksize)
        ax.set_ylabel(name, fontsize=ticksize * 1.5)
        if chroms.shape[0]>1:
            plt.xticks([x for x in chroms.mid], [str(x) for x in chroms.index], rotation=-90, fontsize=ticksize * 1.5)
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        plt.locator_params(axis='y', nbins=4)
        mpl.rc('ytick', labelsize=ticksize)
        if ylim is not None:    plt.ylim(ymin=ylim)
    chroms = pd.DataFrame(df.groupby('CHROM').POS.apply(lambda x:x.max()-x.min()).rename('len').loc[df.reset_index().CHROM.unique()] + 1000)
    chroms = pd.DataFrame(df.groupby('CHROM').POS.apply(lambda x:x.max()).rename('len').loc[df.reset_index().CHROM.unique()] + 1000)
    if CHROMLen is not None:
        chroms=pd.DataFrame(CHROMLen)
    chroms['offset'] = np.append([0], chroms.len.cumsum().iloc[:-1].values)
    chroms['color'] = [c for (_, c) in zip(range(chroms.shape[0]), cycle(colors))]
    if color is not None: chroms['color']=color
    chroms['start']=df.groupby('CHROM').POS.min()
    if CHROMLen is not None:
        chroms['start']=0

    chroms['mid'] = [x + y / 2 for x, y in zip(chroms.offset+chroms.start, chroms.len)]
    chroms['mid'] = [x + y / 2 for x, y in zip(chroms.offset+chroms.start, chroms.len)]
    df['color'] = chroms.color.loc[df.CHROM].values
    df['gpos'] = df.POS + chroms.offset.loc[df.CHROM].values
    df['color'] = chroms.color.loc[df.CHROM].values
    df.set_index('gpos', inplace=True);

    if shade is not None:
        shade['gstart']=shade.start #
        shade['gend']=shade.end #
        if chroms.shape[0]>1:
            shade['gstart']+= chroms.offset.loc[shade.CHROM].values
            shade['gend']+=+ chroms.offset.loc[shade.CHROM].values
        if 'name' in shade.columns:
            shade.sort_values('gstart',ascending=False,inplace=True)
            shade['ID']=range(1,shade.shape[0]+1)
    addGlobalPOSIndex(common, chroms);
    addGlobalPOSIndex(Outliers, chroms)
    if fig is None and axes is None:
        fig,axes=plt.subplots(columns.size, 1, sharex=True,sharey=shareY,figsize=(20, columns.size * 4));
        if columns.size==1:
            axes=[axes]
    elif axes is None:
        axes=fig.axes

    for i in range(columns.size):
        if not i:
            sh=shade
        else:
            if shade is not None and 'name' in shade.columns:
                sh= shade.drop('name', 1)
        plotOne(df[columns[i]], df.color, names[i], chroms,common, sh,axes[i])
    plt.setp(plt.gca().get_xticklabels(), visible=True)
    xlabel='Chromosome'
    if chroms.shape[0]==1:xlabel+=' {}'.format(chroms.index[0])
    axes[-1].set_xlabel(xlabel, size=ticksize * 1.5)
    plt.gcf().subplots_adjust(bottom=0.2)
    if fname is not None:
        print ('saving ', fname)
        plt.savefig(fname)
    if not show:
        plt.ion()

    return fig



def setSize(ax, fontsize=5):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
    try:
        for item in ([ax.zaxis.label] + ax.get_zticklabels()):
            item.set_fontsize(fontsize)
    except:
        pass




class scan:
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


def addGlobalPOSIndex(df,chroms):
    if df is not None:
        df['gpos'] = df.POS + chroms.offset.loc[df.CHROM].values
        df.set_index('gpos', inplace=True);
        df.sort_index(inplace=True)

