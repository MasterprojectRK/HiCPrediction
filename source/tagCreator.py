from configurations import *

def createSetTag(params):
    tmp = params['cellType'] + '_' + params['resolution']
    tmp +='_'+ createProteinTag(params)
    tmp += '_W' + params['windowOperation']
    tmp += str(params['windowSize'])

    if params['equalize']:
        tmp += '_E'
    if params['ignoreCentromeres']:
        tmp += '_A'
    return tmp + params['chrom']

def createProteinTag(params):
    tmp = ''
    tmp += 'M' +params['mergeOperation']
    if params['normalize']:
        tmp += '_N'
    if params['peakColumn']  != 6:
        tmp += '_PC' + str(params['peakColumn'])
    return tmp


def createModelTag(params):
    tmp = createSetTag(params)
    tmp +=  '_C' + params['conversion']
    tmp +=  '_L' + params['lossfunction']
    return tmp
