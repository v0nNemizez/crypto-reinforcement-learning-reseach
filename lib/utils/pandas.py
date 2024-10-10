import os
import pandas as pd

def open_and_convert_data(path: str):
    data = pd.read_csv(path)
    d = data[['date', 'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']]
    data_sorted = d.sort_values(by='date', ascending=False) 


    return data_sorted

def create_merged_dataframe(path: str) -> pd.DataFrame :
    dfs = []
    print('compiling dataframe...')

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.endswith('.csv'):
            df = pd.read_csv(file_path)
            dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f'Total number of rows after merge: {len(merged_df)}')


    return merged_df