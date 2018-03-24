from preprocessing import *
from data.conect2db import *
from utils.Utils import options
from data.DataLoader import MiniBatchLoader

options_dic = readDictionary("../src/properties/options.properties")
fpath = readDictionary("../src/properties/fpath.properties")

print("loading data set from database ........")
test_dataset = getDatasetfromDB(
    vocabDic_path=fpath['vocabDic_path'],
    properties_path=fpath['properties_path']
)
print("loading successfully!")

cycle_counter = 0
for graph in test_dataset:
    if graph.hasCycle:
        cycle_counter += 1
        print("graph {} has cycle".format(graph.sid))

print("There are {} graph have cycle in total!".format(cycle_counter))