from neuralModel import *

def applyAE(args, test=False): 
    print(args)
    customLogVec = np.vectorize(customLog)
    reverseCustomLogVec = np.vectorize(reverseCustomLog)
    model =SparseAutoencoder(args)
    model.load_state_dict(torch.load(MODEL_D+args.model))
    model.eval()
    if test:
        d = TEST_D
        end = "11038"
    else:
        d = CHUNK_D 
        end = "15238"
    name = args.chrom + "_"+end
    matrix = name +".cool"
    ma = hm.hiCMatrix(d+matrix)
    m = ma.matrix.todense()
    m = np.triu(m)
    m[m < args.treshold] = 0
    print(m[0])
    # print(m.shape)
    newCus = deepcopy(m)
    if args.prepData == 'log':
        m += 1
        m = np.log(m) /  np.log(args.maxValue)
    elif args.prepData == 'customLog':
        m= customLogVec(m)
        newCus= customLogVec(newCus)
    else:
        m = m.tolist()
    print(m[0])
    t = torch.Tensor([[m]])
    encoded, decoded = model(t)
    decoded = decoded[0][0]
    encoded = encoded[0]
    new = decoded.detach().numpy()
    new2 = deepcopy(new)
    print(new[0])
    if args.prepData == 'customLog' or args.prepData == 'log':
        new *= np.log(args.maxValue)
        new = np.exp(new)
        new -= 1
        # en *= np.log(args.maxValue)
        # en = np.exp(en)
        # en -= 1
    if args.prepData == 'customLog':
        new2 = reverseCustomLogVec(new2)
    print(new[0])
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

def startTraining(args):
    model = SparseAutoencoder(args)
    return train(model,args)

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
    parserOpt.add_argument('--hidden', '-hl',type=str, default="S")
    parserOpt.add_argument('--output', '-ol',type=str, default="S")
    parserOpt.add_argument('--outputSize', '-os',type=int, default=200)
    parserOpt.add_argument('--maxValue', '-mv',type=int, default=10068)
    parserOpt.add_argument('--convSize', '-cs',type=int, default=5)
    parserOpt.add_argument('--firstLayer', '-fl',type=int, default=4)
    parserOpt.add_argument('--secondLayer', '-sl',type=int, default=8)
    parserOpt.add_argument('--thirdLayer', '-tl',type=int, default=16)
    parserOpt.add_argument('--dropout1', '-d1',type=float, default=0.1)
    parserOpt.add_argument('--pool1', '-p1',type=int, default=2)
    parserOpt.add_argument('--dimAfterP1', '-dim1',type=int, default=0)
    parserOpt.add_argument('--dimAfterP2', '-dim2',type=int, default=0)
    parserOpt.add_argument('--dropout2', '-d2',type=float, default=0.3)
    parserOpt.add_argument('--padding', '-p',type=int, default=0)
    parserOpt.add_argument('--batchSize', '-b',type=int, default=256)
    parserOpt.add_argument('--model', '-m',type=str, default='autoencoder.pt')
    parserOpt.add_argument('--saveModel', '-sm', default=True)
    parserOpt.add_argument('--loss', '-l',type=str, default='L1')
    parserOpt.add_argument('--prepData', '-dp',type=str, default='log')
    parserOpt.add_argument('--lastLayer', '-ll',type=str, default='third')
    args = parser.parse_args(args)
    args.dimAfterP1 = int(args.cutWidth / args.pool1)
    args.dimAfterP2 = int(args.dimAfterP1 / args.pool1)
    args.padding = int(np.floor(args.convSize / 2)) 
    print(args)
    return args

def main(args=None):
    args = parseArguments(args)
    if args.action == "train":
        startTraining(args)
    elif args.action == "predict":
        applyAE(args, test=True)
        applyAE(args,  test=False)


if __name__ == "__main__":
    # main(sys.argv[1:])
