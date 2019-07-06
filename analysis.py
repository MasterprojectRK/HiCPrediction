from hiCOperations import *
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def createDataset(name):
    cols = ['name','modelScore','r2 score','MSE','MAE','MSLE','AUC', 'window', 'merge',
            'model', 'ep', 'np', 'conversion', 'chrom','trainChroms',
            'loss', 'estimators']
    cols.extend(np.array(list(range(0,201)))/200)
    df = pd.DataFrame(columns=cols)
    df = df.set_index(['name'])
    pickle.dump(df, open(RESULT_D+name, "wb" ) )


def filter():

    c = "19"
    # df = df[df['chrom'] == c]
    # d1 = df[~df['trainChroms'].str.contains(c)]

def heatMap(name, mode=0):
    html = ""
    mode = 0
    
    if mode == 0:
        n = 22
        title = "All"
    elif mode == 1:
        customChroms =[9,11,14,17,19]
        n = 5
        title = "Replication"
    html += "<h1>"+title+"</h1>"
    df = pickle.load(open(RESULT_D+name, "rb" ) )
    df['chrom'] = pd.to_numeric(df['chrom'])
    print(df)
    for co in ['default', 'log','standardLog']:
        if mode == 0:
            corr = pd.DataFrame(columns=range(1,23))
        elif mode == 1:
            corr = pd.DataFrame(columns=customChroms)
        html += "<h2>"+co+"</h2>"
        d = df[df.conversion == co]
        # print(d1['trainChroms'].head(50))
        for v in [False,True]:
            d1 = d[d.trainChroms.str.contains("_") == v ]
            d1 = pd.DataFrame(
                sorted(d1.values, key=lambda x: int(x[13].split('_')[0])),
                columns=d1.columns
            )
            for i, e in d1.iterrows():
                x = e.trainChroms.split("_")
                if mode == 1 and ( len(x) > 1 or int(x[0]) not in
                                customChroms):
                    continue
                if e.trainChroms not in corr.index:
                    corr.loc[e.trainChroms] = np.zeros(n)
                    # corr.loc[e.trainChroms] = np.zeros(22)
                corr.loc[e.trainChroms][e.chrom] = e.AUC
        html += corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(3).render()
    with open(title + ".html", "w") as file:
        file.write(html)

def plots(name):
    n = 10
    df = pickle.load(open(RESULT_D+name, "rb" ) )
    df = df.sort_values(by=['trainChroms', 'chrom', 'conversion'])
    ax = plt.gca()
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True)
    trainList = df.trainChroms.unique()
    tcs = sorted(trainList, key=lambda x: int(x.split('_')[0]))[:n]
    cs = sorted(df.chrom.unique(), key=lambda x: int(x.split('_')[0]))[:n]
    for i, row in enumerate(axes):
        for j, cell in enumerate(row):
            if i == len(axes) - 1:
                cell.set_xlabel("Test: {0:s}".format(cs[j]))
            if j == 0:
                t = tcs[i]
                cell.set_ylabel("{0:s}".format(t))
            d1 = df[df.chrom == cs[j]][df.trainChroms == tcs[i]].iloc[:,16:]
            if len(d1) > 0:
                ax =axes[i,j]
                d1 =d1.T
                d1.plot(ax=ax, legend=False,ylim=(0, 1),xlim=(0, 1))
    plt.legend(["default", "log","standardLog"])
    # plt.tight_layout()
    # plt.xlabel("Test Chroms")
    plt.ylabel("Train Chroms")
    plt.show()

if __name__ == "__main__":
    name = "baseResults.p"
    heatMap(name , 0)
    plots(name)
