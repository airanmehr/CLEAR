'''
Copyleft May 11, 2016 Arya Iranmehr, PhD Student, Bafna Lab, UC San Diego,  Email: airanmehr@gmail.com
'''
from __future__ import print_function

import matplotlib as mpl
import pylab as plt
import seaborn as sns

from UTILS import *
def setStyle(style="darkgrid", lw=2, fontscale=1, fontsize=10):
    sns.axes_style(style)
    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': fontsize});
    mpl.rc('text', usetex=True)
    sns.set_context(font_scale=fontscale, rc={"lines.linewidth": lw})


def get_axis_limits(ax, upper=True):
    return ax.get_xlim()[(0, 1)[upper]], ax.get_ylim()[(0, 1)[upper]]


def getAxRange(ax, axi=0):
    return get_axis_limits(ax, upper=True)[axi] - get_axis_limits(ax, upper=False)[axi]

def getColorMap(n):
    colors = ['darkblue', 'r', 'green', 'darkviolet', 'k', 'darkorange', 'olive', 'darkgrey', 'chocolate', 'rosybrown',
              'gold', 'aqua']
    if n == 1: return [colors[0]]
    if n <= len(colors):
        return colors[:n]
    return [mpl.cm.jet(1. * i / n) for i in range(n)]


def getMarker(n, addDashed=True):
    markers = np.array(['o', '^', 's',  'D', 'd', 'h', '*', 'p','v', '3',  'H', '8','<','2', '4'])[:n]# '<', '>'
    if addDashed: markers = map(lambda x: '--' + x, markers)
    return markers

# mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':30}) ;
# mpl.rc('text', usetex=True)

def addGlobalPOSIndex(df,chroms):
    if df is not None:
        df['gpos'] = df.POS + chroms.offset.loc[df.CHROM].values
        df.set_index('gpos', inplace=True);
        df.sort_index(inplace=True)



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

