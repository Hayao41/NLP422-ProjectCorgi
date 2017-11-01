import src.data_processing.conect2db as c2db


if __name__ == "__main__":

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

    c2db.close(connection)




