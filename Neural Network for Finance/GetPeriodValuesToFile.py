#Imports
import requests
import pandas as pd 
from datetime import datetime

#Symbol and Interval 
symbol = 'BTCUST'
interval = '1w'
"""#Requested URL from binance
url = 'https://api.binance.com/api/v3/klines'
#Parameters for Request
params = {
  'symbol': symbol,
  'interval': interval
}
#Call Request and Get Response.
response = requests.get(url, params=params)
#print(response.json())"""

#Read Data From Binance basic.
url = 'https://api.binance.com/api/v1/klines?symbol='+symbol+'&interval='+interval
print(url)
df = pd.read_json(url)
df.columns = [ "date","open","high","low","close","volume",
    "close time","quote asset volume","number of trades","taker buy base asset volume",
    "Taker buy quote asset volume","ignore"]
df['date'] =  pd.to_datetime(df['date'],dayfirst=True, unit = 'ms')
df.set_index('date',inplace=True)
del df['ignore']

print(df)


#Dataframe to Excel
df.to_excel("/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+interval.upper()+".xlsx",
             sheet_name='Sheet_name_1')  

"""
for json in response.json():
    #Open Time Convert   
    #TimeStamp / 1000  (Divide 1000 to get seconds)
    open_time = datetime.fromtimestamp(int(json[0]/1000)) 
    print("Open Time: ",open_time," - Raw: ", json[0])

print("Interval Count: ",len(response.json()))
"""


#Whats is inside.

"""

#Requested URL from binance
url = 'https://api.binance.com/api/v3/klines'
#Parameters for Request
params = {
  'symbol': 'BTCUSDT',
  'interval': '1w'
}
#Call Request and Get Response.
response = requests.get(url, params=params)
#print(response.json())


    1499040000000,      // Open time
    "0.01634790",       // Open
    "0.80000000",       // High
    "0.01575800",       // Low
    "0.01577100",       // Close
    "148976.11427815",  // Volume
    1499644799999,      // Close time
    "2434.19055334",    // Quote asset volume
    308,                // Number of trades
    "1756.87402397",    // Taker buy base asset volume
    "28.46694368",      // Taker buy quote asset volume
    "17928899.62484339" // Ignore.
"""