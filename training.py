from hiCOperations import *
from tagCreator import createTag

@click.group()
def cli():
    pass

@click.option('--lossfunction', '-lf', default='mse',\
              type=click.Choice(['mse','mae']))
@click.option('--modeltype', '-mt', default='rf',\
              type=click.Choice(['ada', 'rf', 'mlp']))
@click.option('--equalize/--dont-equalize', default=False)
@click.option('--ignoretransarms/--dont-ignoretransarms', default=True)
@click.option('--windowoperation', '-wo', default='avg',\
              type=click.Choice(['avg', 'max', 'sum']))
@click.option('--windowsize', '-ws', default=200)
@click.option('--normalize/--dont-normalize', default=False)
@click.option('--mergeoperation', '-mo', default='avg',\
              type=click.Choice(['avg', 'max']))
@click.option('--cellline', '-cl', default='Gm12878')
@click.option('--conversion', '-co', default='none',\
              type=click.Choice(['standardLog', 'none']))
@click.option('resolution', '-r', default=5000)
@click.option('datasetdir', '-dsd', default='Sets/')
@click.option('modeldir', '-md', default='Models/')
@click.option('setfilepath', '-sfp', default=None)
@click.option('modelfilepath', '-sfp', default=None)
@click.argument('chromosome')
@cli.command()

def train(chromosome, modelfilepath, setfilepath,modeldir,  datasetdir, resolution,\
        conversion, cellline, mergeoperation, normalize, windowoperation,\
        windowsize, ignoretransarms, equalize, modeltype,lossfunction):
    
    if modeltype == "rf":
        model = RandomForestRegressor(max_features='sqrt',random_state=5,\
                    n_estimators =10,n_jobs=4, verbose=2,
                                      criterion=lossfunction)
    elif modeltype ==  "ada":
        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
                  'verbose':True,'learning_rate': 0.01, 'loss': 'ls'}
        model = GradientBoostingRegressor(**params)
    elif modeltype ==  "mlp":
        model = MLPRegressor(hidden_layer_sizes=(20,10,5),verbose=True,
                             early_stopping=True, max_iter=25)
    if setfilepath:
        fileName = setfilepath
    else: 
        setTag =createTag(resolution, cellline, chromosome,\
                    merge=mergeoperation, norm=normalize,\
                    window=windowoperation, eq=equalize, ignore=ignoretransarms)
        fileName = datasetdir + setTag + ".brotli"
    df = pd.read_parquet(fileName)
    df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(value=0)
    X = df[df.columns.difference(['first', 'second','chrom',
                                  'reads'])]
    if conversion == 'none':
        y = df['reads']
    elif conversion == 'standardLog':
        y = np.log(df['reads']+1)

    model.fit(X, y)

    if modelfilepath:
        modelFileName = modelfilepath
    else:
        modelFileName = modeldir + createTag(resolution, cellline,chromosome,merge\
            = mergeoperation,norm=normalize,model=modeltype,loss=lossfunction,\
            window=windowoperation, eq=equalize, ignore=ignoretransarms)+ ".z"
    joblib.dump(model, modelFileName,compress=True ) 


# def predictAll(args):
    # mergeAndSave()
    # shutil.copy(RESULT_D+"baseResults.p",RESULTPART_D + resultName)
    # df = pickle.load(open(RESULT_D+"baseResults.p", "rb" ) )
    # # for cs in [14]:
    # # tcs.extend([])
    # tcs = ["1"]
    # # tcs.extend(["2_4", "1_6","1_10_14","6_10_14"])
    # for cs in tcs:
        # args.chroms = str(cs)
        # for w in ["avg"]:
                # args.windowOperation = w
                # for me in ["avg"]:
                    # args.mergeOperation = me
                    # for m in ["rf"]:
                        # args.model = m
                        # # for p in ["log"]:
                        # for p in ["default", "standardLog", "log"]:
                            # args.conversion = p
                            # for n in [False]:
                                # args.normalizeProteins = n
                                # for e in [False]:
                                    # args.equalizeProteins = e
                                    # if  os.path.isfile(tagCreator(args,"model")):
                                        # model = pickle.load(open( tagCreator(args, "model"), "rb" ) )
                                        # for c in [9]:
                                        # # for c in range(START,END):
                                            # args.chrom = str(c)
                                            # print(tagCreator(args,"pred"))
                                            # if args.directConversion == 0:
                                                # if tagCreator(args,"pred").split("/")[-1] in df.index:
                                                    # print("Exists")
                                                # else:
                                                    # predict(args, model)

                                            # elif  args.directConversion == 1: 
                                                # exists = True

                                                # for suf in ["_A", "_B"]:
                                                    # # args.chrom = str(c) + suf
                                                    # if os.path.isfile(ARM_D +args.chrom+".cool"):
                                                        # if not os.path.isfile(tagCreator(args,"pred")):
                                                            # exists = False
                                                # args.chrom = str(c)
                                                # if exists == False:
                                                    # print(args.chrom)
                                                    # predict(args, model)
                                                # else:
                                                    # print("Both exist")
                                            # else:
                                                # predict(args, model)

# def checkIfAlMetricsExist(args, key):
    # if  not os.path.isfile(tagCreator(args, key)):
        # return False
    # return True

if __name__ == '__main__':
    cli()
