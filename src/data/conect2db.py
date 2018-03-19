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

    try:

        properties = preprocessing.readDictionary(properties_path)
        connection = connect2DB(properties)

        with connection.cursor() as cursor:
            sentence_query = "SELECT * FROM sentence"
            annotation_query = "SELECT relation_root FROM annotation WHERE sid=%s"
            cursor.execute(sentence_query)
            results = cursor.fetchall()
            for result in results:
                print(result['sid'])
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
                print(graph)
                dataset.append(graph)

    except Exception as err:
        print(err)

    except ConnectionError as connerr:
        print(connerr)
        print("[Error] build semantic graph from db failed!")

    else:
        return dataset

    finally:
        connection.close()

