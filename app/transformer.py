import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Provider_Transformer(object):
    
    def __init__(self):
        self.columns = []
    
    def fit(self, X, y=None):
        ohe = OneHotEncoder(categories='auto', drop='if_binary',sparse=False, max_categories=15)
        self.cat_dummy_list=list(X.select_dtypes(['object']).columns)
        ohe_df = pd.DataFrame(ohe.fit_transform(X[self.cat_dummy_list]))
        ohe_df.columns=ohe.get_feature_names_out(self.cat_dummy_list)
        self.columns=ohe_df.columns
        
    def transform(self,X,y=None):
        ohe = OneHotEncoder(categories='auto', drop='if_binary',sparse=False, max_categories=15)
        ohe_df = pd.DataFrame(ohe.fit_transform(X[self.cat_dummy_list]))
        ohe_df.columns=ohe.get_feature_names_out(self.cat_dummy_list)
        df=pd.DataFrame(columns=ohe_df.columns)
        common_col_list=list(set(self.columns).intersection(ohe_df.columns))
        df=pd.concat([df,ohe_df[common_col_list]],axis=0)
        return df.fillna(0)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)