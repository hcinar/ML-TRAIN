#Imports
from binance.client import Client
import pandas as pd


#KeySecretPath --> key,secret
key_file_path = r'/home/hus/GIT/ML-TRAIN/Neural Network for Finance/key-secret.txt'


#STATIC FIELDS
key, secret = '',''
#Paste value for internal use :)
#key, secret = 'xxxx','yyyyy'


#SYMBOL
symbol = "BTCUSDT"
from_date = "1 Jan, 2017"


#Export Path Daily and Weekly
weekly_data_export_path = r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+"W"+'.xlsx'
daily_data_export_path = r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+"D"+'.xlsx'


#Check Client Condition
def CheckClient(_client):
    #get client condition #200 OK.
    client_condition = False
    if str(_client.response).lower().count('200') > 0 :
        client_condition = True
        response_time = _client.get_server_time()
        print("Response Time:",response_time)
    if not client_condition:
        print("Server gets an connection error.\n Response:",_client.response)
        quit()

#Get keys from file
def GetKeysFromFile(key_file_path):
    with open(key_file_path) as f:
        line = f.readline()
        key,secret = line.split(',')
    return key,secret

#Get weekly data from file
def GetWeeklyDataToExcel(symbol, from_date, data_export_path, client):
    # fetch weekly klines since it from_Date into excel file
    weekly = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1WEEK, from_date)
    df = pd.DataFrame(weekly)
    df.to_excel(data_export_path)
    return weekly

#Get daily data from file
def GetDailyDataToExcel(symbol, from_date, data_export_path, client):
    # fetch daily klines since it from_Date into excel file
    daily = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, from_date)
    df = pd.DataFrame(daily)
    df.to_excel(data_export_path)
    return daily


if __name__ == "__main__":
    #Check key and secret.
    if key == '' or secret == '':
        key, secret = GetKeysFromFile(key_file_path)

    #Client Access
    client = Client(key, secret)
    
    #Connect and Get Data
    CheckClient(client)
    GetWeeklyDataToExcel(symbol, from_date, weekly_data_export_path, client)
    GetDailyDataToExcel(symbol, from_date, daily_data_export_path, client)