#!/usr/bin/env python
# =============================================================================
# @file   ApplyCuts.py
# @author C. Marin Benito (carla.marin.benito@cern.ch)
# @date   18.06.2015
# =============================================================================
"""Apply cuts to a TTree and save it to a new file"""

# imports
import argparse, os
import ROOT
from ROOT import TFile, TTree

# definition of functions for this script
def getCuts(cutsFile):
    """
    Read cuts from a .txt file and return a string
    Cuts on different lines in the file are concatenated
    with an && string.
    :cutsFile: type str. Name of the file that contains the cuts
    """
    with open(cutsFile, 'r') as f:
        cuts = f.readlines()
        cuts = [c.strip() for c in cuts]
        cut_string = " && ".join(cuts)
    print("cuts read from %s" %cutsFile)
    return cut_string

def applyCuts(fileName, treeName, cuts, newName="_seletion"):
    """
    This function applies cuts to a given TTree and saves
    the resulting one to a new root file.
    Returns the name of the new root file (str) and the efficiency
    of the applied cut (float). It also creates a text file where
    the original file, the applied selection and the resulting
    efficiency are written.
    Description of the arguments:
    :fileName: type str
    Path to the root file with the TTree to cut
    :treeName: type str
    Name of the TTree inside the root file. It can include
    directories, ex: "Bd2KstGamma/DecayTree"
    :cuts: type str
    A single string with the set of cuts to be applied to
    the TTree. The name of the TTree branches and C++ syntax
    should be used, ex: "Kplus_IPCHI2 > 16 && B_MM < 7000"
    :newName: type str
    Suffix to be added to the original file name to create
    the new one. Default: "_selection" will create an
    "originalName_selection.root" output file
    """
    
    # read tree
    file = TFile(fileName)
    tree = file.Get(treeName)
    entries = tree.GetEntries()
    
    # open new file
    newFileName = fileName.replace(".root", "%s.root" %newName)
    newFile = TFile(newFileName, "recreate")
    
    # cut tree and write it
    cutTree = tree.CopyTree(cuts)
    passEntries = float(cutTree.GetEntries())
    cutTree.Write()
    newFile.Close()
    
    # compute and write efficiency
    eff = passEntries/entries

    print("%s candidates in the initial tuple"    % entries)
    print("%s candidates pass the selection"      % passEntries)
    print("The efficiency of the selection is %s" % eff)

    textFile = "selection%s.txt" %newName
    with open(textFile, 'w') as output:
        output.write("Selection: \n %s\n" %cuts)
        output.write("File:      \n %s\n" %fileName)
        output.write("Efficiency:\n %s\n" %eff)
    
    return newFileName, eff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", action="store", type=str, help="file name"       )
    parser.add_argument("cuts", action="store", type=str, help="cuts to apply"   )
    parser.add_argument("-t", "--tree", default="DecayTree" , action="store", type=str, help="tree name (def: DecayTree)"          )
    parser.add_argument("-n", "--name", default="_selection", action="store", type=str, help="sufix for new file (def: _selection)")
    args = parser.parse_args()
    fileName = args.file
    treeName = args.tree
    cutsFile = args.cuts
    newFName = args.name

    # Read cuts from file if needed
    if cutsFile.endswith('.txt'):
        if os.path.exists(cutsFile):
            cuts = getCuts(cutsFile)
        else:
            print('ERROR: File %s not found' %cutsFile)
            exit()
    else:
        print("ERROR: cannot parse file %s" %cutsFile)
        print("Valid formats are: .txt")
        exit()


    newFileName, eff = applyCuts(fileName, treeName, cuts, newFName)
    
#EOF
