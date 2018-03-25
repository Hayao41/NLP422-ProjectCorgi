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
                    dataset.append(graph)
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


def data_load_test(vocabDic_path, properties_path):

    sids = []
    records = []
    # sids.append("S#12258")
    # sids.append("S#2361")
    # sids.append("S#23776")
    # sids.append("S#32686")
    # sids.append("S#32943")
    # sids.append("S#6767")
    sids.append("S#9148")
    sids.append("S#27009")
    sids.append("S#13604")

    vocabDics = preprocessing.loadVocabDic(["pos", "rel"], vocabDic_path)
    pos2idx = vocabDics["pos"]
    rel2idx = vocabDics["rel"]
    dataset = []
    cycle_counter = 0

    try:

        properties = preprocessing.readDictionary(properties_path)
        connection = connect2DB(properties)

        with connection.cursor() as cursor:
            sentence_query = "SELECT * FROM sentence WHERE sid=%s"
            annotation_query = "SELECT relation_root FROM annotation WHERE sid=%s"
            for sid in sids:
                cursor.execute(sentence_query, (sid,))
                results = cursor.fetchall()
                for result in results:
                    print("#============building semantic graph [{}]============#".format(sid))
                    cursor.execute(annotation_query, (sid,))
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
                        sid=sid
                    )

                    if graph.hasCycle:
                        cycle_counter += 1
                        records.append(sid)

                    else:
                        print(graph)
                        dataset.append(graph)

        print("There are {} graphs has cycle then drop them from dataset!".format(cycle_counter))
        print(records)

    except Exception as err:
        print(err)

    except ConnectionError as connerr:
        print(connerr)
        print("[Error] build semantic graph from db failed!")

    else:
        return dataset

    finally:
        connection.close()


