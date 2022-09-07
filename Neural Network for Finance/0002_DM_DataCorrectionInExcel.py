#Imports
import pandas as pd

#SYMBOL
symbol = "BTCUSDT"


#Data Import Paths 
data_import_paths= [ r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+"W"+'.xlsx',
                     r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+"D"+'.xlsx']


#Data Export Paths
data_export_paths = [r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+"W"+'CHG'+'.xlsx',
                     r"/home/hus/GIT/ML-TRAIN/Neural Network for Finance/"+symbol+"D"+'CHG'+'.xlsx']

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


if __name__ == "__main__":
    #Read excel from paths
    for path_index,i_path in enumerate(data_import_paths):

        #Read excel from path
        df = pd.read_excel(i_path)

        ColumnNameCorrection(df)
        ColumnDeletion(df)
        RowTimeFormatCorrection(df)

        #Export to excel file.
        e_path = data_export_paths[int(path_index)]
        df.to_excel(e_path,index=False)

        #Give info when completed.
        print(i_path,"\n Exported Into-----> ",e_path)
