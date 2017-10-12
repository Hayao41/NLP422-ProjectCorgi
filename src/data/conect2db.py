import pymysql


def connection():
    # 打开数据库连接
    db = pymysql.connect("localhost", "root", "rootpass1994326", "test_dataset")
    return db


def close(conection):
    conection.close

