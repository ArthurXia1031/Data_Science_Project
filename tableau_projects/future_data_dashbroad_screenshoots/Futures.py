# -*- coding:utf-8 -*-
# @Time       :6/27/23 & 1:13 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains


import requests
import time
import pymysql

url = "https://open-api.coinglass.com/public/v2/open_interest_history?symbol=BTC&time_type=all&currency=USD"

headers = {
    "accept": "application/json",
    "coinglassSecret": "9c4956a9eb3a461ea14503024a35bcbe"
}

futures = requests.get(url, headers=headers).json()

futures = dict(futures['data'])

dateList = futures['dateList']
priceList = futures['priceList']
dataMap = futures['dataMap']

# datetime

for n in range(len(dateList) - 1):
    timeStamp = dateList[n] / 1000
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    dateList[n] = otherStyleTime


class Mysql:
    # database, table name
    def __init__(self, db_name):
        self.mydb = pymysql.connect(
            host='localhost',
            user='root',
            password='password',
            database=db_name,
        )
        self.cursor = self.mydb.cursor()

    def __del__(self):
        self.cursor.close()
        self.mydb.close()

    def commit(self, query):
        print(query)
        try:
            self.cursor.execute(query)
            self.mydb.commit()
        except Exception as e:
            print(e)


for key, values in dataMap.items():
    for n in range(len(values)):
        if values[n] is None:
            values[n] = 0

# MySQL
Mysql = Mysql('BN')


# 这个api 可以获得 即时的 交易数据，但做图表不需要即时数据
# This api can get the real-time trading data, but the chart does not need real-time data
# 这里取了前一天的数据。
# Here is the data from the previous day.

# insert into database

# 第一次跑 跑一遍历史数据 即可
# The first time you run it, you can run through the historical data
# 之后只用更新最新数据
# Only update the latest data later

def initialize():
    query = """create table if not exists futures_volume (ds datetime, exchange varchar(100), volume float, btc_price float)"""

    Mysql.commit(query)

    for n in range(len(dateList) - 1):
        ds = dateList[n]
        price = priceList[n]
        for exchange in dataMap.keys():
            vol = dataMap[exchange][n]
            # print(ds, price, exchange, vol)
            query = "insert into futures_volume (ds, exchange, volume, btc_price) values ('%s', " % ds + "'%s', " % exchange + "%f, " % vol + "%f)" % price
            Mysql.commit(query)


# initialize()

# 更新昨日数据
# update yesterday data
def daily_update():
    # yesterday
    n = len(dateList) - 2

    ds = dateList[n]
    price = priceList[n]
    for exchange in dataMap.keys():
        vol = dataMap[exchange][n]
        query = "insert into futures_volume (ds, exchange, volume, btc_price) values ('%s', " % ds + "'%s', " % exchange + "%f, " % vol + "%f)" % price
        # print(query)
        Mysql.commit(query)

# 更新昨日数据
# update yesterday data
# daily_update()
