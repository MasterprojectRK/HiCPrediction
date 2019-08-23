import pytest
from source.configurations import *
from source.createBaseFile import loadAllProteins, getProteinFiles, loadProtein,\
        addGenome
from pybedtools.featurefuncs import TSS

@pytest.fixture
def params():
    params = dict()
    params['peakColumn'] = 6
    params['mergeOperation'] = 'avg'
    params['normalize'] = False

    return params

@pytest.fixture
def proteinList():
    path1 = 'TestData/Rad21_1_2.narrowPeak'
    path2 = 'TestData/Ctcf_1_2.narrowPeak'
    proteinFiles = [path1, path2]
    return proteinFiles

def test_getProteinFiles(params, proteinFiles):
    result = getProteinFiles(proteinFiles, params)
    for i in range(len(result[path1])):
        # print(comparePyBed(result[path1][i],pybedtools.BedTool(path1)[i]))
        assert comparePyBed(result[path1][i],pybedtools.BedTool(path1)[i])
    for i in range(len(result[path2])):
        # print(comparePyBed(result[path2][i],pybedtools.BedTool(path2)[i]))
        assert comparePyBed(result[path2][i],pybedtools.BedTool(path2)[i])

def test_getProteinFilesNormalized(params, proteinFiles):
    params['normalize'] = True
    result = getProteinFiles(proteinFiles, params)
    for i in range(len(result[path1])):
        # print(comparePyBed(result[path1][i],pybedtools.BedTool(path1Norm)[i]))
        assert comparePyBed(result[path1][i],pybedtools.BedTool(path1Norm)[i])
    for i in range(len(result[path2])):
        # print(comparePyBed(result[path2][i],pybedtools.BedTool(path2Norm)[i]))
        assert comparePyBed(result[path2][i],pybedtools.BedTool(path2Norm)[i])

def test_addGenome(tmp_path):
    cD = addGenome('TestData/testGenome.cool', str(tmp_path /\
                                "baseFile.ph5"),[1,2], str(tmp_path))
    tD = dict()
    tD['chr1'] =hm.hiCMatrix('TestData/testChr1.cool')
    tD['chr2'] =hm.hiCMatrix('TestData/testChr2.cool')
    dictEqual = {k: tD[k] for k in tD if k in cD and tD[k] == cD[k]}
    assert dictEqual

def test_loadProtein(params, proteinFiles):
    result = getProteinFiles(proteinFiles, params)
    cuts =[0,500] 
    data = loadProtein(result, 'chr1',cuts, params)
    columns = ['start', 0,1]
    dummy = [[0.0,6,7],[500.0,8.5,6.5]] 
    test = pd.DataFrame(dummy,columns=columns, index=range(2))
    assert data.equals(test)

def comparePyBed(a,b):
    for i in range(10):
        print(a[i], b[i])
        if a[i] != b[i]:
            return False
    return True

def create(path, path2):
    a = pybedtools.BedTool(path)
    a1 = a.sort().filter(lambda x: x.chrom == 'chr2' or x.chrom ==
                  'chr1').saveas(path2)
    print(a1)
    # print(a2)
    result = a1

def show():
    path = 'Data/Proteins/Gm12878_Rad21.narrowPeak'
    a = pybedtools.BedTool(path)
    a1 = a.sort().filter(lambda x: x.chrom == 'chr20')
    print(a1)

if __name__ == '__main__':
    params = dict()
    params['peakColumn'] = 6
    params['normalize'] = False
    params['mergeOperation'] = 'avg'
    path1 = 'TestData/Rad21_1_2.narrowPeak'
    path2 = 'TestData/Ctcf_1_2.narrowPeak'
    proteinFiles = [path1, path2]
    test_loadProtein(params, proteinFiles)
