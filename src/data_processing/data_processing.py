import os
import src.data_processing.conect2db as c2db
import nltk
from nltk.parse.stanford import StanfordDependencyParser

if __name__ == "__main__":

    os.environ['STANFORD_PARSER'] = '/Users/joelchen/Desktop/stanford-parser-full-2017-06-09/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = '/Users/joelchen/Desktop/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'

    connection = c2db.connection()

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = connection.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("SELECT * FROM annotation")

    # 使用 fetchone() 方法获取单条数据.
    data = cursor.fetchone()

    print(data)

    cursor.execute("SELECT sentence FROM sentence")

    data = cursor.fetchone()

    parser = StanfordDependencyParser(
        model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
    )

    for parse in parser.raw_parse("I persuade Fred to leave this room."):
        for temp in parse.triples():
            print(temp)

    c2db.close(connection)




