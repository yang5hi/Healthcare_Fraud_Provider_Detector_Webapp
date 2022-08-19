import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler#, MinMaxScaler
from transformer import Provider_Transformer

class Fraud_Detector(object):
    def __init__(self):
        pass
    def build_model(self):
        """
        Building model pipeline 
        """
        steps = [#('preprocessor',preprocessor),
        ('rescale', StandardScaler()),
        #('rescale', MinMaxScaler()),
        ('clf', LogisticRegression(class_weight ='balanced',C=0.01,random_state=67, max_iter=10000))]
        self.pipeline = Pipeline(steps)
    def train(self):
        """
        Train a model 
        """
        # load the data from csv to pandas dataframe
        # X_test_aggregated_raw = pd.read_csv("data/X_test_0817.csv")
        X_train_aggregated_raw = pd.read_csv("data/X_train_0817.csv")
        # y_test_aggregated_raw = pd.read_csv("data/y_test_0817.csv")
        y_train_aggregated_raw = pd.read_csv("data/y_train_0817.csv")

        # Define feature and target 
        target = ["Provider", "PotentialFraud"]
        features = list(X_train_aggregated_raw.columns)
        self.features = [fea for fea in features if fea not in target]

        y_train=y_train_aggregated_raw['PotentialFraud']
        # y_test=y_test_aggregated_raw['PotentialFraud']
        X_train=X_train_aggregated_raw.drop('Provider',axis=1).fillna(0)
        # X_test=X_test_aggregated_raw.drop('Provider',axis=1).fillna(0)

        self.build_model()
        self.model = self.pipeline.fit(X_train, y_train)
    
    def predict(self, context):
        """
        context: dictionary format {'TotalTEDiagCode':502,... etc}
        return np.array
        """
        num_predictions = len([context[self.features[0]]])
        print(num_predictions)
        X = pd.DataFrame(context,index=range(num_predictions))
        return self.model.predict_proba(X)