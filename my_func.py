

### this file contains my functions and other one's that can be used multiply times
import pickle
import pandas as pd
from datetime import datetime, timedelta



def save_model(model = None, features = []):
    name = str(model.__class__).split('.')[-1][:-2] + '_' + datetime.today().strftime("%d%m%Y_%H_%M") + '.pickle'    
    if model:
        with open(name, 'wb') as file:
            pickle.dump((model, features), file)
            print('Save', name)
            
def save_table(table, file_path, table_name):
    name = file_path +'\\' + table_name + '.pickle'
    with open(name, 'wb') as file:
        pickle.dump(table, file)
        print('Save', name)
        
def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        model_tpl = pickle.load(file)
    return model_tpl    

### reduce memory

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# дивимось на категоріальні дані, та їх кількість
# в зміну cat_col_lst зберігаю назви категоріальних колонок
# вивиджу датасет із кількістю унікальних значень в кожній колонці і кількість місінгів

def categorical_col_info(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    подаємо на вхід датафрейм.
    функція відбирає обєкти, і далі видає датафрейм, який 
    показує назву колонки, кількість унікальних значень, і кількість пропущених значень
    """
    
    
    unique_values_lst = []
    missing_values_lst = []
    cat_col_lst = []

    for col in df.select_dtypes(include='object').columns:
        unique_values = df[col].nunique()
        missing_values = df[col].isna().sum()

        cat_col_lst.append(col)
        unique_values_lst.append(unique_values)
        missing_values_lst.append(missing_values)

    column_summary_df = pd.DataFrame({
        'Column': cat_col_lst,
        'Unique values': unique_values_lst,
        'Missing values': missing_values_lst
    })

    column_summary_df.loc['Total'] = [len(cat_col_lst), sum(unique_values_lst), sum(missing_values_lst)]
    return column_summary_df


def count_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values_percent = (missing_values / len(df)) * 100
    result_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Missing Values (%)': missing_values_percent.values
    })
    result_df = result_df[result_df['Missing Values'] > 0]  # Filter out columns with no missing values
    missing_values_df_sorted = result_df.sort_values(by='Missing Values (%)', ascending=False)
    

    return missing_values_df_sorted
