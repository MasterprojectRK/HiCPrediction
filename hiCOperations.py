from configurations import *
from proteinLoad import loadAllProteins
from setCreator import createDataset
from tagCreator import createTag, chromStringToList

def createCombinedDataset(sets):
    data = pd.DataFrame() 
    for df in sets.values():
        data = data.append(df)
    return data


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
    for i in range(1,4):
        print(i)
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
    for cs in [11,14,17,9,19]:
        args.chroms = str(cs)
        for c in ["9_A"]:
            args.chrom = str(c)
            for p in ["default"]:
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

