#Imports
from numpy import zeros
import pandas as pd

#SYMBOL
symbol = "BTCUSDT"


#Data Import Paths 
data_import_paths= [r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+"D"+'.xlsx']


#Data Export Paths
data_export_paths = [r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+"D"+'CHG'+'.xlsx']

#Correct column names
def ColumnNameCorrection(df):
        df.columns=['X',
                'Open time',
                'Open', 
                'High', 
                'Low', 
                'Close', 
                'Volume', 
                'Close time', 
                'Quote asset volume', 
                'Number of trades', 
                'Taker buy base asset volume', 
                'Taker buy quote asset volume', 
                'Ignore']

#Delete unused columns.
def ColumnDeletion(df):
    del df['X']
    del df['Volume']
    del df['Close time']
    del df['Quote asset volume']
    del df['Number of trades']
    del df['Taker buy base asset volume']
    del df['Taker buy quote asset volume']
    del df['Ignore']

#Correction of Open Time format.
def RowTimeFormatCorrection(df):
    df['Open time'] =pd.to_datetime(df['Open time'], unit='ms')


#Correction of Open Time format.
def RowWeeklyCloumnsAndCalculations(df):
        #Set Weekly Open for each 7d before
        df['WOPEN'] = df['Open']
        df.loc[0, 'WOPEN'] = ''
        for i in range(1, len(df)):
            if i - 5 > 0:
                df.loc[i, 'WOPEN'] = df.loc[i-6, 'Open']
            elif i - 5 <= 0:
                df.loc[i, 'WOPEN'] = ''

        #Set Weekly High for each 7d before
        df['WHIGH'] = df['High'].rolling(7).max()

        #Set Weekly Min for each 7d before
        df['WLOW'] = df['Low'].rolling(7).min()

        #Set Weekly Open for each 7d before
        df['WCLOSE'] = df['Close']
        df.loc[0, 'WCLOSE'] =''
        for i in range(1, len(df)):
            if i - 5 > 0:
                df.loc[i, 'WCLOSE'] = df.loc[i-6, 'Close']
            elif i - 5 <= 0:
                df.loc[i, 'WCLOSE'] = ''


if __name__ == "__main__":
    #Read excel from paths
    for path_index,i_path in enumerate(data_import_paths):

        #Read excel from path
        df = pd.read_excel(i_path)

        ColumnNameCorrection(df)
        ColumnDeletion(df)
        RowTimeFormatCorrection(df)
        RowWeeklyCloumnsAndCalculations(df)

        #Export to excel file.
        e_path = data_export_paths[int(path_index)]
        df.to_excel(e_path,index=False)

        #Give info when completed.
        print(i_path,"\n Exported Into-----> ",e_path)
