def cutMatrix(args):
    model = chromsToName(args.chroms)
    targetDir = PRED_D+args.conversion+ "/" +args.chrom +"P/" +model+"/"
    if not os.path.exists(targetDir+"Chunks/"):
        os.mkdir(targetDir+"Chunks/")
    ma = hm.hiCMatrix(targetDir + "matrix.cool")
    matrix = ma.matrix
    matrix = matrix.todense()
    matrix = np.triu(matrix)
    matrix = np.tril(matrix, args.cutWidth-1)
    done = False
    length, width = matrix.shape
    start = 0
    end = args.cutLength
    cuts = ma.cut_intervals
    while(not done):
        print(end)
        if(end >= length):
            end = length - 1
            start = end - args.cutLength
            done = True

        newCuts = cuts[start:end]
        first = newCuts[0][1]
        last = newCuts[-1][2]
        corrCuts = [] 
        for cut in newCuts:
            c = cut[0]
            s = cut[1]
            e = cut[2]
            v = cut[3]
            nCut = (c,s-first, e -first,v)
            corrCuts.append(nCut)
        chunk = matrix[start:end, start:end]
        chunk =  sparse.csr_matrix(chunk)
        region = args.chrom+":"+str(first)+"-"+str(last)
        m = hm.hiCMatrix(None)
        m.setMatrix(chunk,corrCuts)
        region_end =  int(int(region.split("-")[1]) / 10000)
        name = str(region_end).zfill(5) 
        # name = region.split(":")[0] +"_" + str(region_end).zfill(5) 
        m.save(targetDir+"Chunks/"+name+".cool")
        start += args.cutLength - args.overlap
        end += args.cutLength - args.overlap

def iterateAll(args):
    allChunks = []
    for f in os.listdir(CHROM_D):
        args.chrom = ma.getChrNames()[0]
        args.targetDir = CHROM_D+args.chrom+"/"
        cutMatrix(args)

def convertMatrix(hicMa, method):
    if method == "customLog":
        customLogVec = np.vectorize(customLog)
        matrix = hicMa.matrix.todense()
        print(matrix[0])
        matrix = customLogVec(matrix)
        print(matrix[0])
        matrix = np.round(matrix, 2)
        matrix = sparse.csr_matrix(matrix)
        hicMa.setMatrix(matrix, hicMa.cut_intervals)
        return hicMa
    if method =='treshold':
        matrix = hicMa.matrix.todense()
        matrix[matrix < 100] = 10
        # matrix = np.log(matrix) / np.log(20000)
        matrix = sparse.csr_matrix(matrix)
        hicMa.setMatrix(matrix, hicMa.cut_intervals)
        return hicMa
    if method =='manualAssign':
        manualVec = np.vectorize(manualAssign)
        matrix = hicMa.matrix.todense()
        print(matrix[0])
        matrix = manualVec(matrix)
        matrix = sparse.csr_matrix(matrix)
        hicMa.setMatrix(matrix, hicMa.cut_intervals)
        return hicMa
def convertAll(chromosome):
    i = 0
    d = CHUNK_D
    for f in os.listdir(d):
        c = f.split("_")[0]
        e = f.split("_")[1]
        if chromosome == int(c):
        # if "00205.cool" == e
                ma = hm.hiCMatrix(d+f)
                # ma = convertMatrix(ma, 'customLog')
                ma.save(CUSTOM_D  + f)
                i += 1
                if i > 10:
                    break

def reverseCustomLog(a):
    if a == 0:
        return 0
    elif a > reverseMiddle:
        a += sink
        a *= np.log(maxLog)
        a = np.exp(a)
        a += shift
        return a
    else:
        return (a**(1/exponent))*divisor

sink = 0
logMax = 10000
reverseMiddle = 0
shift = 0 #32.5
middle = 0
divisor = 1#50
exponent =1#5

def manualAssign(a):
    if a < 40: return 0
    elif a < 80: return 0.1
    elif a < 150: return 0.2
    elif a < 250: return 0.3
    elif a < 400: return 0.4
    elif a < 600: return 0.5
    elif a < 800: return 0.6
    elif a < 1500: return 0.7 
    elif a < 2500: return 0.8
    else: return 0.9
def customLog(a):
    if a == 0:
        return 0
    elif a >= middle:
        a -= shift
        return np.log(a) /np.log(logMax) - sink
    else:
        return (a/divisor)**exponent
def createDataset(args, create_test =False):
    if create_test:
        d = TEST_D
    else:
        d = CHUNK_D
    matrixList = []
    customLogVec = np.vectorize(customLog)
    for f in os.listdir(d):
        c = f.split("_")[0]
        if args.chrom == c:
            print(f)
            ma = hm.hiCMatrix(d+f)
            matrix = ma.matrix.todense()
            matrix = np.triu(matrix)
            safe = deepcopy(matrix)

            maxV = matrix.max()
            if np.isnan(matrix).any() or maxV == 0:
                continue
            matrix[matrix < args.treshold] = 0
            if args.prepData == 'customLog':
                print(matrix[0])
                matrix = customLogVec(matrix)
                print(matrix[0])
            elif args.prepData == 'log':
                matrix += 1
                matrix = np.log(matrix) /np.log(args.maxValue)
            else: 
                matrix = np.asarray(matrix)
            matrixList.append(matrix)
    else:
        tensor_x = torch.cat(tuple(torch.Tensor([[i]]) for i in matrixList))
    print(tensor_x.shape)
    dataset = utils.TensorDataset(tensor_x)
    test =""
    if create_test:
        test = "_test"
    tre = "_t"+str(args.treshold)
    pickle.dump(dataset, open( SET_D + args.chrom+tre+"_"+args.prepData+test+".p", "wb" ) )
