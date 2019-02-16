from UTILS import *

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



def getRegionPrameter(CHROM,start,end):
    if start is not None and end is not None:CHROM='{}:{}-{}'.format(CHROM,start,end)
    elif start is None and end is not None:CHROM='{}:-{}'.format(CHROM,end)
    elif start is not None and end is None :CHROM='{}:{}-'.format(CHROM,start)
    return CHROM

class VCF:
    @staticmethod
    def loadCHROMLenCDF(PMF=False):
        a=VCF.loadCHROMLen()
        if PMF:
            return (a/a.sum()).round(2)
        return (a.cumsum()/a.sum()).round(2)
    @staticmethod
    def loadCHROMLen(assembly=19,CHROM=None,all=False,autosomal=False):
        if assembly is None:
            return pd.concat([VCF.loadCHROMLen(19), VCF.loadCHROMLen(38)], 1, keys=[19, 38])
        a=pd.read_csv(home + 'storage/Data/Human/ref/hg{}.chrom.sizes'.format(assembly), sep='\t', header=None).applymap(
            lambda x: INT(str(x).replace('chr', ''))).set_index(0)[1]
        if CHROM is not None: a=a.loc[CHROM]
        if not all: a=a.loc[range(1,23)+list('XYM')]
        a.index.name='CHROM'
        if autosomal:
            a=a.loc[range(1,23)]
        return a.rename('len')

    @staticmethod
    def AllPops():
        p = home + 'Kyrgyz/info/kyrgyz.panel'
        return ['1KG']+list(set(VCF.pops(p) + VCF.pops() + VCF.superPops(p) + VCF.superPops()))

    @staticmethod
    def All1KGPops():
        p = '/home/arya/storage/Data/Human/1000GP/info/panel'
        return ['1KG'] + list(set(VCF.pops(p) + VCF.superPops(p) ))

    @staticmethod
    def IDs(P, panel=home + 'POP/HAT/panel', color=None, name=None, maxn=1e6):
        return pd.concat([VCF.ID(p=p,panel=panel,color=color,name=name,maxn=maxn) for p in P])

    @staticmethod
    def IDfly():
        z = pd.read_csv('/home/arya/fly/all/RC/all.folded.gz.col').iloc[1:, 0]
        z.index = pd.MultiIndex.from_tuples(z.apply(lambda x: tuple(map(INT,x.split('.')))), names=['POP', 'GEN', 'REP'])
        return z.sort_index()

    @staticmethod
    def ID(p,panel=home + 'POP/HAT/panel',color=None,name=None,maxn=1e6):
        a = VCF.loadPanel(panel)
        try:a=pd.concat([a, VCF.loadPanel(home + 'Kyrgyz/info/kyrgyz.panel')])
        except: pass
        if p=='1KG':
            x=a.set_index('super_pop').loc[['AFR','EUR','EAS','SAS','AMR']]
        else:
            try:
                x = a.set_index('pop').loc[p]
            except:
                x = a.set_index('super_pop').loc[p]
        x= list(set(x['sample'].tolist()))
        x=pd.Series(x,index=[(name,p)[name is None]] *len(x))
        if color is not None:
            x=x.rename('ID').reset_index().rename(columns={'index':'pop'})
            x['color']=color
        maxn = min(x.shape[0],int(maxn))
        x=x.iloc[:maxn].astype(str)
        x.index.name='pop'
        return x.rename('ID')

    @staticmethod
    def pops(panel=home + 'POP/HAT/panel'):
        return list(VCF.loadPanel(panel)['pop'].unique())
    @staticmethod
    def superPops(panel=home + 'POP/HAT/panel'):
        return list(VCF.loadPanel(panel)['super_pop'].unique())

    @staticmethod
    def getN(panel=home+'/storage/Data/Human/1000GP/info/panel'):
        pan=VCF.loadPanel(panel)
        return pd.concat([pan.groupby('pop').size(),pan.groupby('super_pop').size(),pd.Series({'ALL':pan.shape[0]})])
    @staticmethod
    def getField(fname,field='POS'):
        fields={'CHROM':1,'POS':2,'ID':3}
        cmd="zgrep -v '#' {} | cut -f{}".format(fname,fields[field])
        return pd.Series(Popen([cmd], stdout=PIPE, stdin=PIPE, stderr=STDOUT,shell=True).communicate()[0].strip().split('\n')).astype(int)

    @staticmethod
    def header(fname):
        cmd="zgrep -w '^#CHROM' -m1 {}".format(fname)
        return Popen([cmd], stdout=PIPE, stdin=PIPE, stderr=STDOUT,shell=True).communicate()[0].split('\n')[0].split()
    @staticmethod
    def headerSamples(fname):
        return map(INT,VCF.header(fname)[9:])

    @staticmethod
    def loadPanel(fname=home + 'POP/HAT/panel'):
        return  pd.read_table(fname,sep='\t').dropna(axis=1)

    @staticmethod
    def loadPanels():
        panels = pd.Series({'KGZ': '/home/arya/storage/Data/Human/Kyrgyz/info/kyrgyz.panel',
                           'ALL':  '/home/arya/storage/Data/Human/1000GP/info/panel'})
        load = lambda x: VCF.loadPanel(x).set_index('sample')[['super_pop', 'pop']]
        return pd.concat(map(load, panels.tolist()))

    @staticmethod
    def getDataframeColumns(fin,panel=None,haploid=False):
        def f(x):
            try:return tuple(panel.loc[x].tolist())
            except:return ('NAs','NAp')
        cols=[]
        if panel is not None:
            load=lambda x: VCF.loadPanel(x).set_index('sample')[['super_pop','pop']]
            if isinstance(panel,str): panel=[panel]
            else: panel=panel.tolist()
            panel= pd.concat(map(load,panel))
            try:
                ids=VCF.headerSamples(fin)
                for x in ids:
                    if haploid:
                        cols += [f(x) + (x, 'A')]
                    else:
                        cols += [f(x) + (x, 'A'), f(x) + (x, 'B')]
                cols = pd.MultiIndex.from_tuples(cols, names=['SPOP', 'POP', 'ID', 'HAP'])
            except:
                panel['HAP']='A'
                cols= panel.reset_index().rename(columns={'super_pop':'SPOP','pop':'POP','sample':'ID'}).set_index(['SPOP', 'POP', 'ID', 'HAP']).index
        else:
            for x in VCF.headerSamples(fin):
                cols+=[( x,'A'),(x,'B')]
            cols=pd.MultiIndex.from_tuples(cols,names=[ 'ID','HAP'])
        return cols

    @staticmethod
    def getDataframe(CHROM,start=None,end=None,
                     fin=PATH.OKG+'ALL.chr{}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz',
                     bcftools="/home/arya/bin/bcftools/bcftools",
                     panel=PATH.OKG+'integrated_call_samples_v3.20130502.ALL.panel',haploid=False,dropDots=True,
                     gtfile=False,pop=None,freq=True
                     ):
        reg=getRegionPrameter(CHROM,start,end)

        fin=fin.format(CHROM)
        if freq:

            df= gz.loadFreqGT(f=fin, istr=reg,pop=pop)#.set_index(range(5))
        else:
            cmd="{} filter {} -i \"N_ALT=1 & TYPE='snp'\" -r {} | {} annotate -x INFO,FORMAT,FILTER,QUAL,FORMAT | grep -v '#' | tr '|' '\\t'|  tr '/' '\\t' | cut -f1-5,10-".format(bcftools,fin,reg,bcftools)
            #cmd="{} filter {} -i \"N_ALT=1 & TYPE='snp'\" -r {} | {} annotate -x INFO,FORMAT,FILTER,QUAL,FORMAT | grep -v '#' | cut -f1-5,10-".format(bcftools,fin,reg,bcftools)
            csv=Popen([cmd], stdout=PIPE, stdin=PIPE, stderr=STDOUT,shell=True).communicate()[0].split('\n')
            df = pd.DataFrame(map(lambda x: x.split('\t'),csv)).dropna().set_index(range(5))#.astype(int)


        df.index.names=['CHROM','POS', 'ID', 'REF', 'ALT']

        if freq:
            df=df.rename(pop)
        else:
            df.columns=VCF.getDataframeColumns(fin,panel,haploid)
        dropDots=False
        # if dropDots:df[df=='.']=None;
        # else:df=df.replace({'.':0})

        if not freq:
            if haploid:df=df.replace({'0/0':'0','1/1':'1','0/1':'1'})
            try:df=df.astype(int)
            except:df=df.astype(float)

        return df

    @staticmethod
    def computeFreqs(CHROM,start=None,end=None,
                     fin=PATH.OKG+'ALL.chr{}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz',
                     panel=PATH.OKG+'integrated_call_samples_v3.20130502.ALL.panel',
                     verbose=0,hap=False,genotype=False,haploid=False,gtfile=False,pop=None):


        try:
            if verbose:
                import sys
                print( 'chr{}:{:.1f}-{:.1f}'.format(CHROM,start/1e6,end/1e6)); sys.stdout.flush()
            a=VCF.getDataframe(CHROM,int(start),int(end),fin=fin,panel=panel,haploid=haploid,gtfile=gtfile,pop=pop)
            if pop is not None: return a
            if panel is None:
                return a
            if isinstance(panel,str):
                panel=pd.Series({'HA':panel})
            if panel.size==1:
                a=pd.concat([a],1,keys=panel.index)
            else:
                a=a.T.sort_index()
                a['DS']='ALL';a.loc[['Sick','Healthy'],'DS']='KGZ'
                a=a.set_index('DS',append=True).reorder_levels([4,0,1,2,3]).sort_index().T
            if hap: return a
            elif genotype:
                # print 'aaaa'
                return a.groupby(level=[0,1,2,3],axis=1).sum()
            else:   #compute AF
                if panel is not None:
                    return pd.concat([a.groupby(level=0,axis=1).mean(),a.groupby(level=1,axis=1).mean(),a.groupby(level=[1,2],axis=1).mean()],1)
                else:
                    return a.mean(1).rename('ALL')

        except :
            # print None
            return None

    @staticmethod
    def len(CHROM,ref=19):
        if ref==19:
            a=pd.read_csv(home+'storage/Data/Human/ref/hg19.chrom.sizes',sep='\t',header=None).set_index(0)[1]
            return a.loc['chr'+str(CHROM)]
    @staticmethod
    def batch(CHROM,winSize=1e6,ref=19):
        winSize=int(winSize)
        L=VCF.len(CHROM,ref)
        a=pd.DataFrame(range(0, ceilto(L, winSize), winSize),columns=['start'])
        a['end']=a.start+winSize-1
        a['CHROM']=CHROM
        return a
    @staticmethod
    def computeFreqsChromosome(CHROM,fin,panel,verbose=0,winSize=500000,haplotype=False,genotype=False,save=False,haploid=False,nProc=1,gtfile=False):
        print ("""
        :param CHROM: {}
        :param fin:  {}
        :param panel:  {}
        :param verbose: {}
        :param winSize: {}
        :param haplotype: {}
        :param genotype: {}
        :param save: {}
        :param haploid: {}
        :param nProc: {}
         ***************************************
        """.format(CHROM,fin,panel,verbose,winSize,haplotype,genotype,save,haploid,nProc))
        CHROM=INT(CHROM)
        import vcf
        try:
            L=vcf.Reader(open(fin.format(CHROM), 'r')).contigs['chr{}'.format(CHROM)].length
        except:
            try:
                L=vcf.Reader(open(fin.format(CHROM), 'r')).contigs[str(CHROM)].length
                assert L!=None
            except:
                cmd='zgrep -v "#" {} | cut -f2 | tail -n1'.format(fin.format(CHROM))
                L= int(Popen([cmd], stdout=PIPE, stdin=PIPE, stderr=STDOUT,shell=True).communicate()[0].strip())
        print( 'Converting Chrom {}. ({}, {} Mbp Long)'.format(CHROM,L,int(L/1e6)))
        #a=[VCF.computeFreqs(CHROM,start,end=start+winSize-1,fin=fin,panel=panel,hap=haplotype,genotype=genotype,haploid=haploid,verbose=verbose) for start in xrange(0,ceilto(L,winSize),winSize)]
        args=map(lambda start: (CHROM,start,fin,panel,haplotype,genotype,haploid,verbose,winSize,gtfile), range(0,ceilto(L,winSize),winSize))
        from multiprocessing import Pool
        a=Pool(nProc).map(computeFreqsHelper,args)
        a = intIndex(uniqIndex(pd.concat([x  for x in a if x is not None]),subset=['CHROM','POS']))
        if save:
            if haplotype: suff='.hap'
            elif genotype: suff='.gt'
            else: suff=''
            a.to_pickle(fin.format(CHROM).replace('.vcf.gz','{}.df'.format(suff)))
        return  a


    @staticmethod
    def createGeneticMap(VCFin, chrom,gmpath=PATH.data+'Human/map/GRCh37/plink.chr{}.GRCh37.map',recompute=False):
        if os.path.exists(VCFin+'.map') and not recompute:
            print('map file exist!')
            return
        print ('Computing Genetic Map for ', VCFin)
        gm = pd.read_csv(gmpath.format(chrom), sep='\t', header=None,names=['CHROM','ID','GMAP','POS'])
        df = pd.DataFrame(VCF.getField(VCFin).rename('POS'))
        df['GMAP'] = np.interp(df['POS'].tolist(), gm['POS'].tolist(),gm['GMAP'].tolist())
        df['CHROM']=chrom
        df['ID']='.'
        df[['CHROM','ID','GMAP','POS']].to_csv(VCFin+'.map',sep='\t',header=None,index=None)

    @staticmethod
    def subset(VCFin, pop,panel,chrom,fileSamples=None,recompute=False):
        # print pop
        bcf='/home/arya/bin/bcftools/bcftools'
        assert len(pop)
        if pop=='ALL' or pop is None:return VCFin
        fileVCF=VCFin.replace('.vcf.gz','.{}.vcf.gz'.format(pop))
        if os.path.exists(fileVCF) and not recompute:
            print ('vcf exits!')
            return fileVCF
        print ('Creating a vcf.gz file for individuals of {} population'.format(pop))
        if fileSamples is None:
            fileSamples='{}.{}.chr{}'.format(panel,pop,chrom)
            os.system('grep {} {} | cut -f1 >{}'.format(pop,panel,fileSamples))
        cmd="{} view -S {} {} | {} filter -i \"N_ALT=1 & TYPE='snp'\" -O z -o {}".format(bcf,fileSamples,VCFin,bcf,fileVCF)
        os.system(cmd)
        return fileVCF

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


def gzLoadHelper(args):
    f,p,x=args
    return gz.loadFreqChrom(f=f, p=p, x=x)

class gz:
    @staticmethod
    def CPRA(chrom,f=home+'storage/Data/Human/HLI/GT/bim/CPRA/all.gz',keepCHROM=False):
        cut=(' | cut -f2-','')[keepCHROM]
        a=execute('{}/bin/tabix {} {}'.format(home,f,chrom)+cut)
        a.columns=['POS','REF','ALT']
        return a.set_index('POS')

    @staticmethod
    def loadFly(i, pos=None):
        z = gz.load(i=i, f='/home/arya/fly/all/RC/all.folded.gz')
        z.columns = pd.MultiIndex.from_tuples(map(lambda x: tuple(map(INT, x.split('.'))), z.columns),
                                              names=['POP', 'GEN', 'REP'])
        z = z.loc[i.CHROM]
        if pos is not None:
            z = z.loc[pos]
        return z
    @staticmethod
    def loadAA(f, i, code='linear'):
        a = gz.load(f, i, dropIDREFALT=False)
        cols = ['REF', 'ALT', 'ID']
        aa = a.reset_index(cols)[cols].join(gz.load(f, i, AA=True))
        a = a.reset_index(cols, drop=True)
        a = a[(aa.REF == aa.AA) | (aa.ALT == aa.AA)]
        I = (aa.ALT == aa.AA)

        def fix(a, I, code):
            if code == 'linear': k = 2
            if code == 'freq': k = 1
            a.loc[TI(I)] = k - a.loc[TI(I)]
            return a
        return fix(a, I, code)
    @staticmethod
    def load(f='/home/arya/POP/HA/GT/chr{}.vcf.gz',i=None,istr=None,index=True,dropIDREFALT=True,indvs=None,pop=None,AA=False,CHROMS=None,pad=None):
        if pad is not None:
            def expand(i, pad=500000, left=None, right=None):
                pad = int(pad)
                x = i.copy(True)
                if left is not None: pad = left
                x.start = x.start - pad;
                if right is not None: pad = right
                x.end += pad;
                x.start = max(0, x.start)
                x['len'] = x.end - x.start
                return x
            i=expand(i, pad)
        if CHROMS is not None:return pd.concat(map(lambda x: gz.load(f.format(x)),CHROMS)).sort_index()
        if i is not None:
            try:f=f.format(i.CHROM)
            except:pass
            istr='{}:{}-{}'.format(i.CHROM,i.start,i.end)
        # if istr is not None:
        #     xx=istr.split(':')
            # i=pd.Series({'CHROM': xx[0], 'start':xx[1].split('-')[0], 'end':xx[1].split('-')[1]}).apply(INT)
        if AA: f+='.aa.gz'
        if pop is not None:indvs=VCF.ID(pop)

        try:
            cols = pd.read_csv(f + '.col', header=None)[0]
            if indvs is not None:
                if isinstance(indvs,pd.Series):indvs=indvs.tolist()
                try:
                    colsi= (cols.reset_index().set_index(0).iloc[:, 0].loc[['CHROM','POS','ID','REF','ALT']+indvs]).astype(int).tolist()
                except:
                    colsi = (cols.reset_index().set_index(0).iloc[:, 0].loc[['CHROM', 'POS'] + list(indvs)]).astype(int).tolist()
                cols=cols.iloc[colsi]
            else:cols=cols
        except:
            pass


        try:
            if istr is not None:   cmd='/home/arya/bin/tabix {} {}'.format(f,istr)
            else:               cmd='zcat {} '.format(f)
            if indvs is not None:cmd += ' | cut -f' + ','.join(map(lambda x: str(x+1),colsi))

            a=execute(cmd)
        except:
            # print 'No SNPs in '+istr
            return None
        try:
            try:
                a.columns=cols.sort_index().tolist() ### this is very important, cut,sortys by index
            except:
                a.columns = ['ID','REF','ALT'] + cols.sort_index().tolist()
            if dropIDREFALT:
                if 'ID' in a.columns:
                    a=a.drop(['ID','REF','ALT'],axis=1)
        except:
            pass
        if index:
            if a.shape[1]==3:
                name=0
                if AA:name='AA'
                if 'CHROM' in a.columns:
                    a.CHROM=a.CHROM.apply(INT)
                    a=(a.set_index(['CHROM', 'POS'])).iloc[:,0].rename(name)
                else:
                    a[0] = a[0].apply(INT)
                    a = a.set_index([0, 1]).iloc[:, 0].rename(name)
                    a.index.names = ['CHROM', 'POS']
            else:
                a.CHROM = a.CHROM.apply(INT)
                if 'ID' in a.columns:a = a.set_index(['CHROM', 'POS','ID','REF','ALT'])
                else:a=(a.set_index(['CHROM','POS']))

        if len(a.shape)==1 and indvs is not None: a=a.rename(indvs[0])

        return a

    @staticmethod
    def loadFreqChrom(p, x, f =None):
        fs=['/home/arya/POP/KGZU/GT/AF.gz', '/home/arya/POP/KGZU+ALL/GT/AF.gz', '/home/arya/POP/HAT/GT/AF.gz']
        try:
            for f in fs:
                a = polymorphixDF(pd.DataFrame(gz.load(f=f, indvs=p, istr=x)))
                if  a.shape[0]: break
        except: #single population
            a = polymorphixDF(pd.DataFrame(gz.load(f=f.replace('/HAT/', '/{}/'.format(p)), indvs=p, istr=x)))
        if a.shape[1]==1:a=a.iloc[:,0]
        return a.dropna()
    @staticmethod
    def loadFreqGenome(pop, f='/home/arya/POP/KGZU+ALL/GT/AF.gz', daf=False, nProc=1):
        if daf: f=f.replace('/AF.','/DAF.')
        p=pop
        if isinstance(pop,str):p=[pop]
        CHROMS=map(str,range(1,23))
        if nProc==1:
            return pd.concat(map(lambda x: gz.loadFreqChrom(f=f, p=p, x=x), CHROMS))
        else:
            from multiprocessing import Pool
            pool=Pool(nProc)
            args=map( lambda x: (f,p,x), CHROMS)
            a=pd.concat(pool.map(gzLoadHelper,args))
            pool.terminate()
            return a


    @staticmethod
    def loadFreqGT(i=None, f='/home/arya/POP/HA/GT/chr{}.vcf.gz', istr=None, pop=None, AA=False):
        """
        Loads freq from .gz which is GT file and there should be an n file associatged with it for header
        :param i:
        :param f:
        :return:
        """
        try:
            if AA:
                a = (gz.load(i=i, f=f, istr=istr,dropIDREFALT=False,pop=pop).mean(1)/2).rename(pop)
                a=pd.concat([a.reset_index(['ID','REF','ALT']),gz.load(i=i, f=f, istr=istr,AA=True)],1)
                a = a[(a.AA == a.REF) | (a.AA == a.ALT)]
                I = a.ALT == a.AA
                a=a[pop]
                a[I]=1-a[I]
            else:
                a=gz.load(i=i, f=f, istr=istr, pop=pop,dropIDREFALT=False)
                freq=lambda x: x.mean()/2#(x.mean() / 2).rename(pop)
                nomissing=lambda x: x[x>=0]
                a = a.apply(lambda x: freq(nomissing(x)),1)
            return a
        except:
            return None



    @staticmethod
    def code(A,coding='linear'):
        """
        :param coding: can be
        linear: GT={0,1,2}
        dominant: GT={0,1}
        recessive: GT={0,1}
        het: GT={0,1}
        """
        a=A.copy(True)
        if coding=='linear':
            pass
        elif coding=='dominant':
            a[a>0]=1
        elif coding == 'recessive':
            a[a <= 1] = 0
            a[a > 1] = 1
        elif coding == 'het':
            a[a > 1] = 0
        return a
    @staticmethod
    def GT(vcf,coding='linear'):
        """
        :param vcf: path to vcf file
        :param coding: can be
                linear: GT={0,1,2}hq
                dominant: GT={0,1}
                recessive: GT={0,1}
                het: GT={0,1}
        :return:
        """
        from subprocess import Popen, PIPE
        sh='/home/arya/workspace/bio/Scripts/Bash/VCF/createGTSTDOUT.sh'
        sh2='/home/arya/workspace/bio/Scripts/Bash/VCF/sampleNames.sh'
        from StringIO import  StringIO
        with open(os.devnull, 'w') as FNULL:
            a= pd.read_csv(StringIO(Popen([sh, vcf], stdout=PIPE, stdin=FNULL, stderr=FNULL).communicate()[0]), sep='\t', header=None).set_index([0, 1])
            try:
                cols = pd.read_csv(StringIO(Popen([sh2, vcf], stdout=PIPE, stdin=FNULL, stderr=FNULL).communicate()[0]), sep='\t',header=None)[0].tolist()
                a.columns = cols
            except:
                pass
        a.index.names=['CHROM','POS']

        return gz.code(a,coding)

    @staticmethod
    def save(df,f,index=True):
        import uuid
        mkdir(home+'storage/tmp/')
        tmp=home+'storage/tmp/'+str(uuid.uuid4())
        df.to_csv(tmp,sep='\t',header=None)
        if isinstance(df,pd.DataFrame):pd.Series(df.reset_index().columns).to_csv(f+'.col',sep='\t',index=False)
        os.system(home + 'bin/bgzip -c {0} > {1} &&  rm -f {0}'.format(tmp,f))
        if index:os.system(home + 'bin/tabix -p vcf {} '.format(f))




def createAnnotation(vcf ,db='BDGP5.75',computeSNPEFF=True,ud=0,snpeff_args=''):
    #snps=loadSNPID()
    import subprocess
    fname=vcf.replace('.vcf','.SNPEFF.vcf').replace('.gz','')
    fname=vcf+'.SNPEFF.vcf'
    assert fname!=vcf
    if computeSNPEFF:
        cmd='java -Xmx4g -jar ~/bin/snpEff/snpEff.jar {} -ud {} -s snpeff.html {} {} | cut -f1-8 > {}'.format(snpeff_args,ud,db,vcf,fname)
        # print cmd
        subprocess.call(cmd,shell=True)
        # print 'SNPEFF is Done'
    import vcf
    def saveAnnDataframe(fname,x='ANN'):
        # print(x), fname
        ffields = lambda x: x.strip().replace("'", '').replace('"', '').replace(' >', '')
        vcf_reader = vcf.Reader(open(fname, 'r'))
        csv=fname.replace('SNPEFF.vcf',x+'.csv')
        with open(csv,'w') as fout:
            print >>fout,'\t'.join(['CHROM','POS','REF','ID']+map(ffields,vcf_reader.infos[x].desc.split(':')[1].split('|')))
            for rec in  vcf_reader:
                if x in rec.INFO:
                    for line in map(lambda y:('\t'.join(map(str,[INT(rec.CHROM),rec.POS,rec.REF,rec.ID]+y))),map(lambda ann: ann.split('|') ,rec.INFO[x])):
                        # print line
                        if x=='LOF':
                            line=line.replace('(','').replace(')','')
                        print >>fout, line
        uscols=[range(10),range(6)][x=='LOF']
        df = pd.read_csv(csv, sep='\t', usecols=uscols).set_index(['CHROM', 'POS']).apply(lambda x: x.astype('category'))
        df.to_pickle(csv.replace('.csv','.df'))
        try:
            df=df[['Annotation', 'Annotation_Impact', 'Gene_Name', 'Feature_Type']]
            df.to_pickle(csv.replace('.csv','.sdf'))
            gz.save(df, csv.replace('.csv', '.s.gz'))
        except:
            pass
    saveAnnDataframe(fname,'ANN')
    saveAnnDataframe(fname,'LOF')

class DBSNP():
    def __init__(self,hg):
        self.hg=hg
        self.idx=DBSNP.loadIDX(self.hg)
    @staticmethod
    def loadIDX(hg=37):
        f=home+'storage/Data/Human/dbSNP/151/GRCh{}/byBatch/byBatch.idx.gz'.format(hg)
        if hg==3738:f = home + 'storage/Data/Human/dbSNP/151/GRCh37/noINFO/hg19/1-22XYM/hg38/byBatch/byBatch.idx.gz'
        return pd.read_csv(f, sep='\t',header=None,names=['batch','start','end']).set_index('batch')

    @staticmethod
    def batch(hg,i):
        print( i,int(i))
        f = '/home/ubuntu/storage/Data/Human/dbSNP/151/GRCh{}/byBatch/{:02d}.gz'.format(hg, int(i))
        if hg==3738:
            f = '/home/ubuntu/storage/Data/Human/dbSNP/151/GRCh37/noINFO/hg19/1-22XYM/hg38/byBatch/{:02}.gz'.format(i)
        return pd.read_csv(f, sep='\t', header=None, index_col=0)


    def load(self,a):
        if isinstance(a,list):
            a=pd.Series(list(set(a)))
        if a.dtype!=int:
            a=a.apply(lambda x: int(x[2:]))
        def f( x):
            if x.size > 0:
                return x.rename(0).reset_index().set_index(0).sort_index().join(DBSNP.batch(self.hg,x.name),how='inner').reset_index()
        batches = pd.cut(a.values, [0] + self.idx['end'].tolist(), labels=self.idx.index)
        a.index = batches.tolist()
        b=a.groupby(level=0).apply(f).reset_index()
        if b.shape[0]:
            b=b[range(5)]
            b.columns=['ID', 'CHROM','POS','REF','ALT']
            # b= b..dropna();
            b.CHROM=b.CHROM.apply(INT);b.POS=b.POS.apply(int)
            b=b.drop_duplicates()
            b.ID=b.ID.apply(lambda x: 'rs'+str(x))
            return b.set_index('ID').sort_index()

    @staticmethod
    def loadCAD(risk,assembly):
        dataset=risk.dataset.iloc[0]
        f=home + 'CAD/raw/{}.dbSNP{}.df'.format(dataset,assembly)
        try:
            raise 0
            a=pd.read_pickle(f)
        except:
            ID=risk.ID
            ID=ID[ID.apply(lambda x: x.split(';')[0][:2]=='rs')].apply(lambda x: int(x.split(';')[0][2:]))
            a=DBSNP(assembly).load(ID)
            # a.to_pickle(f)
        return a


def computeFreqsHelper(args):
    CHROM,start,fin,panel,hap,genotype,haploid,verbose,winSize,gtfile=args
    end=start+winSize-1
    return VCF.computeFreqs(CHROM=CHROM,start=start,end=end,fin=fin,panel=panel,verbose=verbose,hap=hap,genotype=genotype,haploid=haploid,gtfile=gtfile)
