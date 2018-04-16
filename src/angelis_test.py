
import numpy as np
import conect2db
import preprocessing


def getMetrics(test_set):

    # metrics
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    p = 0
    r = 0
    F1 = 0
    acc = 0

    for graph in test_set:
        labels = np.array([word.label for word in graph.indexedWords])
        preds = np.array([word.pred for word in graph.indexedWords])

        TP += ((preds == 1) & (labels == 1)).sum()
        TN += ((preds == 0) & (labels == 0)).sum()
        FN += ((preds == 0) & (labels == 1)).sum()
        FP += ((preds == 1) & (labels == 0)).sum()

    # compute metrics
    if TP + FP != 0:
        p = TP / (TP + FP)

    if (TP + FN) != 0:
        r = TP / (TP + FN)

    if (r + p) != 0:
        F1 = 2 * r * p / (r + p)

    acc = (TP + TN) / (TP + TN + FP + FN)

    print("Angeli's Test Data Evaluate Metrics: Test ACC[{:.2f}%] "
          "P[{:.2f}%], R[{:.2f}%], F1[{:.2f}%]\n".format(
            (acc * 100), (p * 100), (r * 100), (F1 * 100)))


if __name__ == "__main__":
    options_dic = preprocessing.readDictionary("../src/properties/options.properties")
    fpath = preprocessing.readDictionary("../src/properties/fpath.properties")

    test_set = conect2db.load_angelis_dataset(
                vocabDic_path=fpath['vocabDic_path'],
                properties_path=fpath['properties_path'],
                test_id_path=fpath['test_id_path']
            )

    getMetrics(test_set)
