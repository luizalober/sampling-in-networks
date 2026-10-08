import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def pca_table_processing(df: pd.DataFrame, components:int=3, drop_columns:int=[])->pd.DataFrame:
    
    df = df.copy()
    
    col_names = df.drop(drop_columns,axis=1).columns

    mm_scaler = MinMaxScaler()
    df_mm = mm_scaler.fit_transform(df.drop(drop_columns, axis = 1))

    df_mm = pd.DataFrame(df_mm, columns=col_names)
    
    pca = PCA(n_components=components)

    X = np.array(df_mm.iloc[:,0:df_mm.shape[1]])
    pca.fit(X)

    evr = np.round(pca.explained_variance_ratio_,3)

    print('Explained Variance Ratio (EVR):\n' )
    for index, value in enumerate(evr):
        print('PCA Component {p} EVR: {v}%'.format(p=(index+1), v=np.round(value*100,3)))
    
    for i in range(components):
        pca_column = pca.transform(X)[:,i]
        column_name = 'pca_component_'+str(i+1)
        df_mm[column_name] = pca_column
    
    for column in drop_columns[::-1]:
        df_mm.insert(loc=0, column=column, value=df[column].copy())
    
    return df_mm