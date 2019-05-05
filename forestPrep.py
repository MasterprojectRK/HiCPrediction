from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
def applyForest(args, test=False): 
    df = pickle.load(open( SET_D+args.chrom+"_allWindows.p", "rb" ) )
    new = sparse.csr_matrix(new)
    newCus = sparse.csr_matrix(newCus)
    new2 = sparse.csr_matrix(new2)
    # plotMatrix(d,matrix)
    print(name)
    ma.setMatrix(new, ma.cut_intervals)
    ma.save(PRED_D + name + "_P_L1.cool")
    plotMatrix(PRED_D, name + "_P_L1.cool", True)
    if args.prepData == 'customLog':
        ma.setMatrix(new2, ma.cut_intervals)
        ma.save(PRED_D + name + "_P_L1_Own.cool")
        plotMatrix(PRED_D, name + "_P_L1_Own.cool", True)
    # ma.setMatrix(newCus, ma.cut_intervals)
    # ma.save(PRED_D + name + "_Cus.cool")
    # plotMatrix(PRED_D, name + "_Cus.cool", True

def splitDataset(args)
    df = pickle.load(open( SET_D+args.chrom+"_allWindows.p", "rb" ) )
    train, test = train_test_split(dataset, test_size=0.2)
    pickle.dump(train, open( SET_D +args.chrom+"_train.p", "wb" ) )
    pickle.dump(test, open( SET_D +args.chrom+"_test.p", "wb" ) )

def startTraining(args):
    model = RandomForestRegressor(random_state=0,n_estimators=20, verbose=3)
    df = pickle.load(open( SET_D+args.chrom+"_train.p", "rb" ) )
    train_X = df[df.columns.difference(['first, second, target'])]
    train_y = df['target']
    model.fit(X,y)
    pickle.dump(model, open( MODEL_D +args.chrom+"_model.p", "wb" ) )
    
def parseArguments(args=None):
    print(args)

    parser = argparse.ArgumentParser(description='HiC Prediction')

    parserRequired = parser.add_argument_group('Required arguments')

    # define the arguments
    parserRequired.add_argument('--action', '-a', choices=['train',
         'predict'], help='Action to take', required=True)

    parserOpt = parser.add_argument_group('Optional arguments')
    parserOpt.add_argument('--learningRate', '-lr',type=float, default=0.001)
    parserOpt.add_argument('--epochs', '-e',type=int, default=1000)
    parserOpt.add_argument('--treshold', '-tr',type=int, default=0)
    parserOpt.add_argument('--beta', '-be',type=float, default=0.1)
    parserOpt.add_argument('--chrom', '-c',type=str, default="4")
    parserOpt.add_argument('--cutLength', '-cl',type=int, default=50)
    parserOpt.add_argument('--maxValue', '-mv',type=int, default=10068)
    parserOpt.add_argument('--batchSize', '-b',type=int, default=256)
    parserOpt.add_argument('--model', '-m',type=str, default='autoencoder.pt')
    parserOpt.add_argument('--saveModel', '-sm', default=True)
    parserOpt.add_argument('--loss', '-l',type=str, default='L1')
    parserOpt.add_argument('--prepData', '-dp',type=str, default='log')
    parserOpt.add_argument('--validationData', '-v',type=bool, default=False)
    parserOpt.add_argument('--trainData', '-t',type=bool,default=True)
    args = parser.parse_args(args)
    print(args)
    return args

def main(args=None):
    args = parseArguments(args)
    if args.action == "train":
        startTraining(args)
    elif args.action == "predict":
        predict(args)
        predict(args)
    elif args.action == "split":
        splitDataset(args)


if __name__ == "__main__":
    main(sys.argv[1:])
