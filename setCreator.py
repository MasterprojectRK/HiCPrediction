from configurations import *
from tagCreator import tagCreator

def createForestDataset(args, allProteins=None):
    start = time.time()*1000
    if allProteins == None:
        allProteins = pickle.load(open(tagCreator(args, "protein"), "rb" )).values.tolist()
    colNr = np.shape(allProteins)[1]
    zeroProteins = np.zeros(colNr -1)
    rows = np.shape(allProteins)[0]
    print(rows)
    end = time.time()*1000
    print("Time2: %d" % (end - start))
    tmp = args.chrom
    args.chrom = args.chrom.split("_")[0]
    ma2 = pickle.load(open(tagCreator(args, "matrix"), "rb" ))
    args.chrom = tmp
    ma = pickle.load(open(tagCreator(args, "matrix"), "rb" ))
    end = time.time()*1000
    print("Time3: %d" % (end - start))
    reads = ma.todense()
    logs = deepcopy(reads)
    norms = deepcopy(reads)
    maxValue = np.max(ma2.todense()) 
    logs += 1
    logs = np.log(logs+1)
    logs /= np.log(maxValue)
    norms =norms /  maxValue
    cols = ['first', 'second','chrom'] + list(range(3*
         (colNr-1)))+['distance','reads','logTarget','normTarget']
    window = []
    if args.windowOperation == 'avg':
        convertF = np.mean
    elif args.windowOperation == 'sum':
        convertF = np.sum
    elif args.windowOperation == 'max':
        convertF = np.max
    rows = 1000
    end = time.time()*1000
    print("Time4: %d" % (end - start))
    start = time.time()*1000
    for frame in rowGenerator(args, rows, allProteins, reads, logs, norms):
        window.append(frame)
    end = time.time()*1000
    print("Time5: %d" % (end - start))
    print(len(window))
    start = time.time()*1000
    window = []
    for j in range(rows):
        maxReach = min(int(args.reach)+1,rows-j)
        firstStart = allProteins[j][0]
        for  i in range(0,maxReach):
            if j+1 >= j+i:
                middleProteins = zeroProteins
            else:
                middleProteins = convertF(allProteins[j+1:j+i], axis=0)[1:]
            frame = [firstStart, allProteins[j+i][0],args.chrom]
            frame.extend(allProteins[j][1:])
            frame.extend(middleProteins)
            frame.extend(allProteins[j+i][1:])
            frame.extend([i*args.binSize, reads[j,j+i],logs[j,j+i],norms[j,j+i]])
            window.append(frame)
    end = time.time()*1000
    print("Time6: %d" % (end - start))
    print(len(window))
    # data = pd.DataFrame(window,columns =cols)
    # print(data.shape)
    # pickle.dump(data, open(tagCreator(args, "set"), "wb" ) )

def rowGenerator(args, rows, allProteins, reads, logs, norms):
    colNr = np.shape(allProteins)[1]
    zeroProteins = np.zeros(colNr -1)
    if args.windowOperation == 'avg':
        convertF = np.mean
    elif args.windowOperation == 'sum':
        convertF = np.sum
    elif args.windowOperation == 'max':
        convertF = np.max
    for j in range(rows):
        maxReach = min(int(args.reach)+1,rows-j)
        firstStart = allProteins[j][0]
        for  i in range(0,maxReach):
            if j+1 >= j+i:
                middleProteins = zeroProteins
            else:
                middleProteins = convertF(allProteins[j+1:j+i], axis=0)[1:]
            frame = [firstStart, allProteins[j+i][0],args.chrom]
            frame.extend(allProteins[j][1:])
            frame.extend(middleProteins)
            frame.extend(allProteins[j+i][1:])
            frame.extend([i*args.binSize, reads[j,j+i],logs[j,j+i],norms[j,j+i]])
            yield frame

def createAllCombinations(args):
    for w in ["avg"]:
        args.windowOperation = w
        for m in ["avg"]:
            args.mergeOperation = m
            for n in [False]:
                args.normalizeProteins = n
                for e in [False]:
                    args.equalizeProteins = e
                    createAllWindows(args)


def createAllWindows(args):
    i = 1
    for f in os.listdir(ARM_D):
        args.chrom = f.split(".")[0]
        if  os.path.isfile(tagCreator(args, "protein")):
            # if int(args.chrom.split("_")[0]) < 22:
                # continue
            print(args.chrom)
            if True: 
            # if  not os.path.isfile(tagCreator(args, "set")):
                createForestDataset(args)
