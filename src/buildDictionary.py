
from preprocessing import buildAndSaveVocabDic
from utils import Vocabulary

pos_list = Vocabulary.POS_VOCBULARY
rel_list = Vocabulary.RELATION_VOCBULARY
buildAndSaveVocabDic(pos_list, "pos", "/Users/joelchen/PycharmProjects/NLP422-ProjectCorgi/src/vocabDic/")
buildAndSaveVocabDic(rel_list, "rel", "/Users/joelchen/PycharmProjects/NLP422-ProjectCorgi/src/vocabDic/")
