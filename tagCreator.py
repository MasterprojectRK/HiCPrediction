from configurations import *


def tagCreator(args, mode):
    if args.equalizeProteins:
        ep = "_E"
    else:
        ep = ""
    if args.normalizeProteins:
        nop = "_N"
    else:
        nop = ""
    
    wmep = "_"+args.windowOperation +"_M"+args.mergeOperation+ep +nop

    cowmep =  "_"+args.conversion +  wmep
    csa = chromsToName(args.chroms)+"_"+args.arms
    csam = csa + "_"+args.model +"_" + args.loss

    if mode == "set":
        return SET_D + args.chrom + wmep +".p"

    elif mode == "model":
        return MODEL_D + csam + cowmep+".p"

    elif mode == "cut":
        return CUT_D + args.chrom + ".p"

    elif mode == "matrix":
        return MATRIX_D + args.chrom + ".p"

    elif mode == "protein":
        return PROTEIN_D + args.chrom+ "_M"+args.mergeOperation+nop+".p"

    elif mode == "pred":
        return PRED_D + args.chrom + "_P"+ csam + cowmep +".cool"

    elif mode == "setC":
        return SETC_D+csa+ wmep +".p"

    elif mode == "image":
        return IMAGE_D + args.chrom +"_R" + args.region+"_P"+csam + cowmep +".png"

    elif mode == "plot":
        return PLOT_D  + args.chrom +"_P"+csam + cowmep +".png"

def chromsToName(s):
    return(chromListToString(chromStringToList(s)) )

def chromStringToList(s):
    chroms = []
    parts = s.split("_")
    for part in parts:
        elems = part.split("-")
        if len(elems) == 2:
            chroms.extend(range(int(elems[0]), int(elems[1])+1))
        elif len(elems) == 1:
            chroms.append(int(elems[0]))
        else:
            print("FAAAAAAAAAAAAAAAAAAAAAAAAAIIIIIIIIILLLLLLLLL")
    chroms = list(map(str, chroms))
    return chroms

def chromListToString(l):
    return ("_").join(l)

