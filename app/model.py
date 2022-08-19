import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from transformer import Cost_Transformer

class ToyModel(object):
    def __init__(self):
        pass
    def build_model(self):
        """
        Building model pipeline 
        """
        steps = [('ctf', Cost_Transformer()),
         ('rescale', MinMaxScaler()),
         ('logr', LogisticRegression())]
        self.pipeline = Pipeline(steps)
    def train(self):
        """
        Train a model 
        """
        df = pd.read_csv('data/sales.csv')
        df.dropna(subset=['price'], inplace=True)
        
        features = list(df.columns)
        
        # Define feature and target 
        target = ["price", "luxury"]
        self.features = [fea for fea in features if fea not in target]

        X_train = df[self.features]
        y_train = df["price"].map(lambda x: 1 if float(x.strip("$").replace(",", "")) > 500000 else 0)
        self.build_model()
        self.model = self.pipeline.fit(X_train, y_train)
    
    def predict(self, context):
        """
        context: dictionary format {'cost':'$300k'... etc}
        return np.array
        """
        num_predictions = len(context[self.features[0]])
        X = pd.DataFrame(context,index=range(num_predictions))
        return self.model.predict_proba(X)