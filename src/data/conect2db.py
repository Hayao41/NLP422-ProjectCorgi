import pymysql
import preprocessing


def connect2DB(properties):
    
    # connected to data base
    connection = pymysql.connect(
        host=properties['server'],
        user=properties['user'],
        password=properties['pass'],
        db=properties['database'],
        # charset=properties['charset'],
        cursorclass=pymysql.cursors.DictCursor
    )
    if connection:
        print("Connect to data base{} successfully!".format(properties['database']))
    
    return connection


def getDatasetfromDB():
    vocabDics = preprocessing.loadVocabDic(["pos", "rel", "act"], "/Users/joelchen/PycharmProjects/NLP422-ProjectCorgi/src/vocabDic/")
    pos2idx = vocabDics["pos"]
    rel2idx = vocabDics["rel"]
    label2idx = vocabDics["act"]
    dataset = []

    properties = {
        'server': 'localhost',
        'user': 'root',
        'pass': 'rootpass1994326',
        'database': 'test_dataset',
        'charset': 'uft8'
    }

    connection = connect2DB(properties)
    try:
        with connection.cursor() as cursor:
            sentence_query = "SELECT * FROM sentence"
            annotation_query = "SELECT relation_root FROM annotation WHERE sid=%s"
            cursor.execute(sentence_query)
            results = cursor.fetchall()
            for result in results:
                cursor.execute(annotation_query, (result['sid'],))
                annotations = cursor.fetchall()
                if len(annotations) == 0:
                    annotations = [{'relation_root': 0}]

                listLabel = set([annotation['relation_root'] for annotation in annotations])

                graph = preprocessing.buildSemanticGraph(
                    DependencyTreeStr=result['dpTreeStr'],
                    listLabel=listLabel,
                    use_word=False,
                    pos2idx=pos2idx,
                    rel2idx=rel2idx
                )
                dataset.append(graph)

    finally:
       connection.close()
       return dataset

