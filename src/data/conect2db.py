
import os
import pymysql
import preprocessing


def connect2DB(properties):

    ''' 
    connect to data base \n
    @Param:
    properties: data base properties contains \n
    {
        'server' : server name,
        'user' : user name,
        'pass' : password,
        'database' : data base name
        'charset' : encoding charset
    }
    '''
    
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
    else:
        raise ConnectionError("Connect to data base{} failed!".format(properties['database']))
    
    return connection


def getDatasetfromDB(vocabDic_path, properties_path):
    ''' 
    connect to data base then build dataset 
    '''

    vocabDics = preprocessing.loadVocabDic(["pos", "rel"], vocabDic_path)
    pos2idx = vocabDics["pos"]
    rel2idx = vocabDics["rel"]
    dataset = []
    cycle_counter = 0

    try:

        properties = preprocessing.readDictionary(properties_path)
        connection = connect2DB(properties)

        with connection.cursor() as cursor:
            sentence_query = "SELECT * FROM sentence"
            annotation_query = "SELECT relation_root FROM annotation WHERE sid=%s"
            cursor.execute(sentence_query)
            results = cursor.fetchall()
            for result in results:
                print("#============building semantic graph [{}]============#".format(result['sid']))
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
                    rel2idx=rel2idx,
                    sid=result['sid']
                )

                if graph.hasCycle:
                    cycle_counter += 1

                else:
                    print(graph)
                    dataset.append(preprocessing.DataTuple(indexedWords=graph.indexedWords, graph=graph))
        print("There are {} graphs has cycle then drop them from dataset!".format(cycle_counter))

    except Exception as err:
        print(err)

    except ConnectionError as connerr:
        print(connerr)
        print("[Error] build semantic graph from db failed!")

    else:
        return dataset

    finally:
        connection.close()


def splited_load_dataset(vocabDic_path, properties_path, test_id_path):
    
    vocabDics = preprocessing.loadVocabDic(["pos", "rel"], vocabDic_path)
    pos2idx = vocabDics["pos"]
    rel2idx = vocabDics["rel"]
    training_set = []
    test_set = []
    test_set_id = []
    cycle_counter = 0

    if not os.path.exists(test_id_path):
        raise Exception(test_id_path + " did not exit, please change to full data load mode!")

    with open(test_id_path, mode="r", encoding="utf-8") as t_id:
        t_id.readline()
        for line in t_id.readlines():
            if str(line) != "\n":
                test_set_id.append(str(line).split("\n")[0])

    try:

        properties = preprocessing.readDictionary(properties_path)
        connection = connect2DB(properties)

        with connection.cursor() as cursor:
            sentence_query = "SELECT * FROM sentence"
            annotation_query = "SELECT relation_root FROM annotation WHERE sid=%s"
            cursor.execute(sentence_query)
            results = cursor.fetchall()
            for result in results:
                print("#============building semantic graph [{}]============#".format(result['sid']))
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
                    rel2idx=rel2idx,
                    sid=result['sid']
                )

                if graph.hasCycle:
                    cycle_counter += 1

                else:
                    print(graph)
                    if result["sid"] in test_set_id:
                        test_set.append(preprocessing.DataTuple(indexedWords=graph.indexedWords, graph=graph))
                    else:
                        training_set.append(preprocessing.DataTuple(indexedWords=graph.indexedWords, graph=graph))
        print("There are {} graphs has cycle then drop them from dataset!".format(cycle_counter))

    except Exception as err:
        print(err)

    except ConnectionError as connerr:
        print(connerr)
        print("[Error] build semantic graph from db failed!")

    else:
        return training_set, test_set

    finally:
        connection.close()


def data_load_test(vocabDic_path, properties_path):

    vocabDics = preprocessing.loadVocabDic(["pos", "rel"], vocabDic_path)
    pos2idx = vocabDics["pos"]
    rel2idx = vocabDics["rel"]
    dataset = []
    cycle_counter = 0

    try:

        properties = preprocessing.readDictionary(properties_path)
        connection = connect2DB(properties)

        with connection.cursor() as cursor:
            sentence_query = "SELECT * FROM sentence limit 2001"
            annotation_query = "SELECT relation_root FROM annotation WHERE sid=%s"
            cursor.execute(sentence_query)
            results = cursor.fetchall()
            for result in results:
                print("#============building semantic graph [{}]============#".format(result['sid']))
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
                    rel2idx=rel2idx,
                    sid=result['sid']
                )

                if graph.hasCycle:
                    cycle_counter += 1

                else:
                    print(graph)
                    dataset.append(preprocessing.DataTuple(indexedWords=graph.indexedWords, graph=graph))
        print("There are {} graphs has cycle then drop them from dataset!".format(cycle_counter))

    except Exception as err:
        print(err)

    except ConnectionError as connerr:
        print(connerr)
        print("[Error] build semantic graph from db failed!")

    else:
        return dataset

    finally:
        connection.close()


def load_angelis_dataset(vocabDic_path, properties_path, test_id_path):
    vocabDics = preprocessing.loadVocabDic(["pos", "rel"], vocabDic_path)
    pos2idx = vocabDics["pos"]
    rel2idx = vocabDics["rel"]
    test_set = []
    test_set_id = []
    cycle_counter = 0

    if not os.path.exists(test_id_path):
        raise Exception(test_id_path + " did not exit, please change to full data load mode!")

    with open(test_id_path, mode="r", encoding="utf-8") as t_id:
        t_id.readline()
        for line in t_id.readlines():
            if str(line) != "\n":
                test_set_id.append(str(line).split("\n")[0])

    try:

        properties = preprocessing.readDictionary(properties_path)
        connection = connect2DB(properties)

        with connection.cursor() as cursor:
            sentence_query = "SELECT * FROM sentence"
            annotation_query = "SELECT relation_root FROM annotation WHERE sid=%s"
            pred_query = "SELECT relation_root FROM angelis_preds WHERE sid=%s"
            cursor.execute(sentence_query)
            results = cursor.fetchall()
            for result in results:
                print("#============building semantic graph [{}]============#".format(result['sid']))
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
                    rel2idx=rel2idx,
                    sid=result['sid']
                )

                if graph.hasCycle:
                    cycle_counter += 1

                else:
                    print(graph)
                    if result["sid"] in test_set_id:
                        cursor.execute(pred_query, result['sid'])
                        preds = cursor.fetchall()
                        list_pred = set([pred['relation_root'] for pred in preds])
                        for word in graph.indexedWords:
                            if word.sentence_index in list_pred:
                                word.pred = 1

                        test_set.append(preprocessing.DataTuple(indexedWords=graph.indexedWords, graph=graph))

        print("There are {} graphs has cycle then drop them from dataset!".format(cycle_counter))

    except Exception as err:
        print(err)

    except ConnectionError as connerr:
        print(connerr)
        print("[Error] build semantic graph from db failed!")

    else:
        return test_set

    finally:
        connection.close()
