from configurations import *
from proteinLoad import loadAllProteins
from setCreator import createAllWindows, createAllCombinations
from tagCreator import tagCreator



def predictionPreparation(args, pred=1):
    if pred == 1:
        print(tagCreator(args, "pred"))
        pred = pickle.load(open(tagCreator(args, "pred"), "rb" ) )
    if len(args.chrom.split("_")) > 1:
        predictionToMatrix(args, pred)
    else:
        print(tagCreator(args, "pred"))
        c = args.chrom
        args.chrom = c +"_A"
        predA = pred.loc[pred['chrom'] == args.chrom]
        if  os.path.isfile(ARM_D +args.chrom+".cool")\
            and not os.path.isfile(tagCreator(args,"pred")):
            predictionToMatrix(args, predA)
        else: 
            print("A exists")
        
        # args.chrom = c +"_B"
        # predB = pred.loc[pred['chrom'] == args.chrom]
        # if  os.path.isfile(ARM_D +args.chrom+".cool")\
            # and not os.path.isfile(tagCreator(args,"pred")):
            # predictionToMatrix(args, predB)
        # else: 
            # print("B exists")
            



#### NICE CODE SNIPPET###
    # tqdm.pandas()
    # pred.progress_apply(lambda row: convertEntry(new,
                                                 # row['idx1'],row['idx2'],
                                                 # row['predConv']), axis=1)
def predictionToMatrix(args, pred):
    print(pred.head())
    print("Now Converting:")
    ma = hm.hiCMatrix(ARM_D +args.chrom+".cool")
    mat = ma.matrix.todense()
    factor =  np.max(mat)
    if args.conversion == "norm":
        convert = lambda val: val  * factor
    elif args.conversion == "log":
        convert = lambda val: np.exp(val  * np.log(factor)) - 1
    elif args.conversion == "standardLog":
        convert = lambda val: np.exp(val) - 1
    elif args.conversion == "default":
        convert = lambda val: val
    new = sparse.csr_matrix(new)
    pred['idx1'] = pred.index.codes[0]
    pred['idx2'] = pred.index.codes[1]
    pred['predConv'] = convert(pred['pred'])
    data = np.array(pred['predConv'])
    row = np.array(pred['idx1'])
    col = np.array(pred['idx2'])
    new = sparse.csr_matrix((data, (row, col)), shape=mat.shape)
    ma.setMatrix(new, ma.cut_intervals)
    ma.save(tagCreator(args, "pred"))

def storeMatrixAndCuts(args):

    for f in os.listdir(ARM_D):
            args.chrom = f.split(".")[0] 
        # if  not os.path.isfile(tagCreator(args, "matrix")) or\
            # not os.path.isfile(tagCreator(args, "cut")):
            print(f)
            m = hm.hiCMatrix(ARM_D+f)
            matrix = m.matrix
            cuts = m.cut_intervals
            pickle.dump(cuts, open(tagCreator(args, "cut"), "wb" ) )
            pickle.dump(matrix, open(tagCreator(args, "matrix"), "wb" ) )

    for f in os.listdir(CHROM_D):
            args.chrom = f.split(".")[0]
        # if  not os.path.isfile(tagCreator(args, "matrix")) or\
            # not os.path.isfile(tagCreator(args, "cut")):
            print(f)
            m = hm.hiCMatrix(CHROM_D+f)
            matrix = m.matrix
            cuts = m.cut_intervals
            pickle.dump(cuts, open(tagCreator(args, "cut"), "wb" ) )
            pickle.dump(matrix, open(tagCreator(args, "matrix"), "wb" ) )

def divideIntoArms(args):
    ma = hm.hiCMatrix(CHROM_D +args.chrom+".cool")

    f=open("Data2e/BaseData/centromeres.txt", "r")
    fl =f.readlines()
    elems = None
    for x in fl:
        elems = x.split("\t")
        if elems[1] == "chr"+args.chrom:
            print(elems)
            continue
    start = int(elems[2])
    end = int(elems[3])
    cuts = ma.cut_intervals
    i = 0
    cuts1 = []
    cuts2 = []
    print(cuts[4510:4530])
    firstIndex = 0
    for cut in cuts:
        
        if cut[2] < start:
            cuts1.append(cut)
            lastIndex = i + 1
        elif cut[1] > end:
            cuts2.append(cut)
        else:
            firstIndex = i + 1
        i += 1
    if firstIndex == 0:
        firstIndex = lastIndex
    print(len(cuts))
    print(len(cuts1))
    print(len(cuts2))

    m1 = ma.matrix.todense()
    m2 = ma.matrix.todense()
    m1 = m1[:lastIndex,:lastIndex]
    new = sparse.csr_matrix(m1)
    ma.setMatrix(new, cuts1)
    ma.save(ARM_D + args.chrom + "_A.cool")
    
    m2 = m2[firstIndex:,firstIndex:]
    new = sparse.csr_matrix(m2)
    ma.setMatrix(new, cuts2)
    ma.save(ARM_D + args.chrom + "_B.cool")

def createAllArms(args):
    for i in range(1,13):
        args.chrom = str(i)
        start = time.time()
        divideIntoArms(args)
        end = time.time()
        print("Time: %d" % (end - start))
    for i in range(13,16):
        args.chrom = str(i)
        copyfile(CHROM_D +args.chrom +".cool" ,ARM_D + args.chrom + "_A.cool")
    for i in range(16,22):
        args.chrom = str(i)
        start = time.time()
        divideIntoArms(args)
        end = time.time()
        print("Time: %d" % (end - start))
    args.chrom = str(22)
    copyfile(CHROM_D +args.chrom +".cool" ,ARM_D + args.chrom + "_A.cool")

def score(feature):
    return feature.score

def plotPredMatrix(args):

    name = tagCreator(args, "pred")  
    a = ["--matrix",name,
            "--dpi", "300"]
    if args.log:
        a.extend(["--log1p", "--vMin" ,"1"])
    else:
        a.extend(["--vMax" ,"1","--vMin" ,"0"])
    if args.region:
        a.extend(["--region", args.region])
    elif args.regionIndex1 and args.regionIndex2:
        ma = hm.hiCMatrix(name)
        cuts = ma.cut_intervals
        args.region = args.chrom.split("_")[0] +":"+str(cuts[args.regionIndex1][1])+"-"+ str(cuts[args.regionIndex2][1])
        a.extend(["--region", args.region])

    a.extend( ["-out",tagCreator(args, "image")])
    print(a)
    args.region = None
    hicPlot.main(a)

def plotMatrix(args):
    for i in range(5):
        args.regionIndex1 = i*500 + 1
        args.regionIndex2 = (i+1)*500
        name = args.sourceFile.split(".")[0].split("/")[-1]
        a = ["--matrix",args.sourceFile,
                "--dpi", "300"]
        if args.log:
            a.extend(["--log1p", "--vMin" ,"1","--vMax" ,"1000"])
        else:
            a.extend(["--vMax" ,"1","--vMin" ,"0"])
        if args.region:
            a.extend(["--region", args.region])
            name = name + "_r"+args.region
        elif args.regionIndex1 and args.regionIndex2:
            ma = hm.hiCMatrix(args.sourceFile)
            cuts = ma.cut_intervals
            region = args.chrom +":"+str(cuts[args.regionIndex1][1])+"-"+ str(cuts[args.regionIndex2][1])
            a.extend(["--region", region])
            name = name + "_R"+region

        a.extend( ["-out", IMAGE_D+name+".png"])
        hicPlot.main(a)


def plotDir(args):
    for cs in [1,11,14,17,"1_6"]:
        args.chroms = str(cs)
        for c in ["9_A"]:
            args.chrom = str(c)
            for p in ["log","default", "standardLog"]:
                args.conversion = p
                for w in ["avg"]:
                    args.windowOperation = w
                    for me in ["avg"]:
                        args.mergeOperation = me
                        for m in ["rf"]:
                            args.model = m
                            for n in [False]:
                                args.normalizeProteins = n
                                for n in [False]:
                                    args.equalizeProteins = n
                                    if os.path.isfile(tagCreator(args,"pred")):
                                        for i in range(5):
                                            args.regionIndex1 = i*500 + 1
                                            args.regionIndex2 = (i+1)*500
                                            plotPredMatrix(args)


def concatResults():
    sets = []
    for a in os.listdir(RESULTPART_D):
        if a.split("_")[0] == "part":
            if os.path.isfile(RESULTPART_D + a):
                sets.append(pickle.load(open(RESULTPART_D + a, "rb" ) ))
                print(len(pickle.load(open(RESULTPART_D + a, "rb" ) )))
    sets.append(pickle.load(open(RESULT_D+"baseResults.p", "rb" ) ))
    df_all = pd.concat(sets)
    df_all = df_all.drop_duplicates()
    print(len(df_all))
    # df_all = df_all[~df_all.index.duplicated()]
    print(df_all[df_all.index.duplicated()])
    print(len(df_all))

    pickle.dump(df_all, open(RESULT_D+"baseResults.p", "wb" ) )

def mergeAndSave():
    d = RESULT_D
    now = str(datetime.datetime.now())[:19]
    now = now.replace(":","_").replace(" ", "")
    src_dir = d + "baseResults.p"
    dst_dir = d + "/Old/old"+str(now)+".p"
    shutil.copy(src_dir,dst_dir)
    concatResults()

