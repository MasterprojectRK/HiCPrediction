from hiCOperations import *


def plotGraphics(args):
    plt.plot(indices, values)
    plt.savefig(tagCreator(args, "plot"))



def startTraining(args, df):
    if args.model == "rf":
        model = RandomForestRegressor(max_features='sqrt',random_state=5,\
                    n_estimators =args.estimators,n_jobs=4, verbose=2, criterion=args.loss)
    elif args.model ==  "ada":
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'verbose':True,'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    elif args.model ==  "mlp":
        model = MLPRegressor(hidden_layer_sizes=(20,10,5),verbose=True,
                             early_stopping=True, max_iter=25)
    df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(value=0)
    X = df[df.columns.difference(['first', 'second','chrom',
                                  'reads'])]
    if args.conversion == 'default':
        y = df['reads']
    elif args.conversion == 'standardLog':
        y = np.log(df['reads']+1)

    model.fit(X, y)
    pickle.dump(model, open(tagCreator(args, "model"), "wb" ) ) 


def predictAll(args):
    mergeAndSave()
    shutil.copy(RESULT_D+"baseResults.p",RESULTPART_D + resultName)
    df = pickle.load(open(RESULT_D+"baseResults.p", "rb" ) )
    # for cs in [14]:
    # tcs.extend([])
    tcs = ["1"]
    # tcs.extend(["2_4", "1_6","1_10_14","6_10_14"])
    for cs in tcs:
        args.chroms = str(cs)
        for w in ["avg"]:
                args.windowOperation = w
                for me in ["avg"]:
                    args.mergeOperation = me
                    for m in ["rf"]:
                        args.model = m
                        # for p in ["log"]:
                        for p in ["default", "standardLog", "log"]:
                            args.conversion = p
                            for n in [False]:
                                args.normalizeProteins = n
                                for e in [False]:
                                    args.equalizeProteins = e
                                    if  os.path.isfile(tagCreator(args,"model")):
                                        model = pickle.load(open( tagCreator(args, "model"), "rb" ) )
                                        for c in [9]:
                                        # for c in range(START,END):
                                            args.chrom = str(c)
                                            print(tagCreator(args,"pred"))
                                            if args.directConversion == 0:
                                                if tagCreator(args,"pred").split("/")[-1] in df.index:
                                                    print("Exists")
                                                else:
                                                    predict(args, model)

                                            elif  args.directConversion == 1: 
                                                exists = True

                                                for suf in ["_A", "_B"]:
                                                    args.chrom = str(c) + suf
                                                    if os.path.isfile(ARM_D +args.chrom+".cool"):
                                                        if not os.path.isfile(tagCreator(args,"pred")):
                                                            exists = False
                                                args.chrom = str(c)
                                                if exists == False:
                                                    print(args.chrom)
                                                    predict(args, model)
                                                else:
                                                    print("Both exist")
                                            else:
                                                predict(args, model)

def checkIfAlMetricsExist(args, key):
    if  not os.path.isfile(tagCreator(args, key)):
        return False
    return True

