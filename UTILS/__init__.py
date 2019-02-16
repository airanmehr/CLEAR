'''
Copyleft May 23, 2016 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
import os
import numpy as np
import pandas as pd
from subprocess import Popen, PIPE, STDOUT

def mkdir(path):os.system('mkdir -p {}'.format(path))
parentdir=lambda path:os.path.abspath(os.path.join(path, os.pardir))
home = os.path.expanduser('~') + '/'

def INT(x):
    try: return int(x)
    except: return x

def TI(a):
    return a.replace({False:None}).dropna().index

def floorto(x, base=50000):
    return int(base * np.floor(float(x)/base))

def roundto(x, base=50000):
    return int(base * np.round(float(x)/base))

def ceilto(x, base=50000):
    return int(base * np.ceil(float(x)/base))

class path:
    def __init__(self,home):
        self.paper = home + 'workspace/timeseries_paper/'
        self.data = home + 'storage/Data/'
        self.scan = self.data + 'Human/scan/'
        self.Dmel = self.data + 'Dmelanogaster/'
        self.OKG = self.data + 'Human/20130502/ALL/'
        self.paperFigures = self.paper + 'figures/'
        self.plot = home + 'out/plots/';
        self.out = home + 'out/';
        self.simout = self.data + 'SimulationOutFiles/'
        self.stdout = self.out + 'std/'
        self.UKBB = home + '/processed/genetics/imputed/hg38/QCed/'
        mkdir(self.out)
        mkdir(self.simout)
        mkdir(self.plot)
        mkdir(self.stdout)

PATH=path(home)

dedup=lambda x: x[~x.index.duplicated()]

import numbers
def renameColumns(DF,suffix,pre=True):
    df=DF.copy(True)
    if pre:
        df.columns=map(lambda x:'{}{}'.format(suffix,x),df.columns)
    else:
        df.columns=map(lambda x:'{}{}'.format(x,suffix),df.columns)
    return df

def isNumber(x):
    return isinstance(x, numbers.Number)

def convertToIntStr(x):
    if isNumber(x):
        return '{:.0f}'.format(x)
    else:
        return x


def to_hdf5(filename, df, metadf=None, **kwargs):
    store = pd.HDFStore(filename)
    store.put('data', df)
    if metadf is not None:
        store.put('meta', metadf)
    store.get_storer('data').attrs.metadata = kwargs
    store.close()

def read_hdf5(filename):
    with pd.HDFStore(filename) as store:
        data = store['data']
        metadf=None
        try:
            metadf = store['meta']
        except:
            pass
        metadata = store.get_storer('data').attrs.metadata
    return data, metadf, metadata


def CDF(a,round2=2):
    try:
        x= a.round(round2).value_counts().sort_index().cumsum()
        return x/x.iloc[-1]
    except:
        x = a.value_counts().cumsum()
        return x / x.iloc[-1]

def CDFCounts(a,round2=2):
    try:
        x= a.round(round2).value_counts().sort_index().cumsum()
        return x
    except:
        x = a.value_counts().cumsum()
        return x

def CDFPDF(a,round2=2):
    return pd.concat([CDF(a,round2),PMF(a,round2),PMFCounts(a,round2),CDFCounts(a,round2)],1,keys=['CMF','PMF','Mass','CummulativeMass'])


def PMF(a,round2=2):
    try:
        x= a.round(round2).value_counts().sort_index()
        return x/x.sum()
    except:
        # print 'Categorical'
        x = a.value_counts()
        return x / x.sum()

def PMFCounts(a,round2=2):
    try:
        return a.round(round2).value_counts().sort_index()
    except:
        # print 'Categorical'
        return a.value_counts()

def getGeneList( x):  return pd.DataFrame(x.tolist()).stack().unique().tolist()


def intIndex(df):
    names=df.index.names
    df=df.reset_index()
    df[names]=df[names].applymap(INT)
    return df.set_index(names).sort_index()

def uniqIndex(df,keep=False,subset=['CHROM','POS']): #keep can be first,last,None
    names=df.index.names
    if subset is None: subset=names
    return df.reset_index().drop_duplicates(subset=subset,keep=keep).set_index(names).sort_index()

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def execute(cmd,returnDF=True,verbose= False, sep='\t',header=None,escapechar=None):
    if verbose:print( cmd)
    cmd= [cmd]
    with open(os.devnull, 'w') as FNULL:
        if not returnDF: return Popen(cmd, stdout=PIPE, stdin=FNULL, stderr=FNULL, shell=True).communicate()
        return pd.read_csv(StringIO(Popen(cmd, stdout=PIPE, stdin=FNULL, stderr=FNULL,shell=True) .communicate()[0]),sep=sep, header=header,escapechar=escapechar)

        # a=Popen([cmd], stdout=PIPE, stdin=FNULL, stderr=FNULL,shell=True) .communicate()[0]
    # if returnDF: return pd.read_csv(StringIO(a), sep='\t',header=None)



def MAF(y,t=None):
    x = y.copy(True)
    if t is not None:
        x[x[t] > 0.5] = 1 - x[x[t] > 0.5]
    else:
        x[x>0.5]=1-x[x>0.5]
    return x
def polymorphixDF(a,MAF=1e-15):
    if len(a.shape)==1:
        a=pd.DataFrame(a)
    return a[polymorphix(a.abs().mean(1),MAF,True)]
def polymorphix(x, MAF=1e-9,index=False):
    I=(x>=MAF)&(x<=1-MAF)
    if index: return I
    return x[I]
def polymorphic(data, minAF=1e-9,mincoverage=10,index=True):
    def poly(x):return (x>=minAF)&(x<=1-minAF)
    C,D=data.xs('C',level='READ',axis=1),data.xs('D',level='READ',axis=1)
    I=(C.sum(1)/D.sum(1)).apply(lambda x:poly(x)) & ((D>=mincoverage).mean(1)==1)
    if index:
        return I
    return data[I]

def files(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]




def batch(iterable, n=10000000):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]




