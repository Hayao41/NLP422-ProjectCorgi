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

# test_dataset = data_load_test(
#     vocabDic_path=fpath['vocabDic_path'],
#     properties_path=fpath['properties_path']
# )
print("loading successfully!")
