from transformer import Provider_Transformer
from model import Fraud_Detector
import pickle


model = Fraud_Detector()
model.train()
with open('models/model.pkl', 'wb') as f:
	pickle.dump(model,f)