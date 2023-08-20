# -*- coding:utf-8 -*-
# @Time       :6/25/23 & 9:43 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains


"""

每日早晨定时跑一遍即可
重复跑会使数据重复
或者 设置primary key 避免重复数据
使用之前需要将两个历史数据 
csv倒入数据库 
建好两张table
1. CEX_Volume_share   CEX每日SPOT交易量数据 
      只选取了前top 13 兼顾知名度和交易量的 13-14家交易所
2. BTC_Price        每日 BTC 价格

"""

# You can use this code to get the daily volume of the top 13-14 exchanges
# The data is from https://www.coingecko.com/en/exchanges

from urllib.request import urlopen
import urllib
from bs4 import BeautifulSoup
import ssl
import re
from urllib.request import urlopen
from urllib.error import HTTPError
from urllib.error import URLError
import pymysql
import warnings
import datetime

warnings.filterwarnings('ignore')

ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://www.coingecko.com/exchanges'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3775.400 QQBrowser/10.6.4208.400'}
url = urllib.request.Request(url, headers=headers)
response = urllib.request.urlopen(url)
html = response.read().decode("utf-8")

try:
    html
except HTTPError as e:
    print(e)
except URLError as e:
    print('html:rFailed')
    exit()
else:
    print('html:Worked')

bs = BeautifulSoup(html, 'html.parser')

# get the volume
volume = bs.findAll('div', {'class': 'trade-vol-amount text-right'})

vol_list = []

# add the volume to the vol_list
for i in volume:
    vol_list.append(i.get_text().split()[0][1:])

# get the exchange name
ex_name = bs.findAll('span', {'class': 'pt-2 flex-column'})

ex_list = []

for i in ex_name:
    ex_list.append(i.get_text().split()[0])

# correct the names
n = False
for i in range(len(ex_list) - 1):
    if ex_list[i] == 'Binance':
        if n:
            ex_list[i] = 'Binance_US'
        else:
            n = True

    if ex_list[i] == 'MEXC':
        ex_list[i] = 'MEXC_Global'

top_list = ['Binance', 'Coinbase', 'OKX', 'KuCoin', 'Gate.io', 'Huobi', 'Kraken', 'Binance_US', 'Bybit', 'Bitfinex',
            'Gemini', 'MEXC_Global', 'Bitget', 'Bitstamp']


# check the namelist
def checklist(top_list, ex_list):
    n = 1
    for i in ex_list:
        if i in top_list:
            n += 1

    if n == len(top_list):
        return True
    else:
        return False


# get the wrong name
def wrongdetail(top_list, ex_list):
    for i in top_list:
        if i not in ex_list:
            return i


def checkall():
    if checklist(top_list, ex_list):
        print('namelist unfull and lack of: ' + wrongdetail(top_list, ex_list))
        exit()

    else:
        print('namelist check: FULL')


checkall()

# get the btc price

url = 'https://www.coingecko.com/en'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3775.400 QQBrowser/10.6.4208.400'}
url = urllib.request.Request(url, headers=headers)
response = urllib.request.urlopen(url)
html = response.read().decode("utf-8")

bs = BeautifulSoup(html, 'html.parser')
price = bs.findAll('span', {'class': 'no-wrap'})

for i in price:
    bp = float(i.get_text().split()[0][1:].replace(',', ''))
    break

# get the pair record

vol_record = {}

for i in top_list:
    n = 0
    for j in ex_list:
        if i == j:
            vol_record[i] = round(float(vol_list[2 * (n + 1) - 2]) * bp, 1)
        n += 1

print(vol_record)

# write into the database

# get date
totay_date = datetime.date.today()


# print(totay_date)


class Mysql:
    # database, table name
    def __init__(self, db_name):
        self.mydb = pymysql.connect(
            host='localhost',
            user='root',
            password='xeqmysql',
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

    def insert(self, dt, tb, col1, col2, col3):
        for k, v in dt.items():
            query = 'INSERT %s' % tb + ' (%s, ' % col1 + '%s, ' % col2 + '%s)' % col3 + \
                    ' VALUES (' + "'%s'" % k + ',%f' % v + ",'%s'" % totay_date + ')'
            self.commit(query)


Mysql = Mysql('BN')

Mysql.insert(vol_record, 'CEX_Volume_share', 'Exchange', 'Volume', 'ds')

# insert the daily BTC Price
query = "INSERT BTC_Price (Date, Close) VALUES ('%s'" % totay_date + ",%f)" % bp
Mysql.commit(query)
