from transformer import DateTransform, AgeTransform,CodeCountTransform,CodeFrequencyGroupTransform
from transformer import Top15OneHotTransform,ProviderLevelAggregateTransform
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler#, MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from model import Fraud_Detector
import pickle
# load the data from csv to pandas dataframe
provider = pd.read_csv("data/Train-1542865627584.csv")
beneficiary = pd.read_csv("data/Train_Beneficiarydata-1542865627584.csv")
inpatient = pd.read_csv("data/Train_Inpatientdata-1542865627584.csv")
outpatient = pd.read_csv("data/Train_Outpatientdata-1542865627584.csv")

provider['PotentialFraud']=provider['PotentialFraud'].map(lambda x:1 if (x=='Yes' or x==1) else 0)
inpatient['In_Out']=1
outpatient['In_Out']=0

# union/concat the inpatient and outpatient data
concat_df=pd.concat([inpatient, outpatient],axis=0)
merge_bene_df=concat_df.merge(beneficiary, on='BeneID', how='left')
merge_provider_df=merge_bene_df.merge(provider, on = 'Provider', how ='left')


diagnosis_code_columns = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 
           'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10' ]
procedure_code_columns = ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
       'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6']


train_top_15_diag_codes = ['4019', '25000', '2724', 'V5869', '4011', '42731', 'V5861', '2720', '2449',
 '4280', '53081', '41401', '496', '2859', '41400', 'Other']
train_top_15_proc_codes = ['4019.0', '9904.0', '2724.0', '8154.0', '66.0', '3893.0', '3995.0', '4516.0',
 '3722.0', '8151.0', '8872.0', '9671.0','4513.0','5849.0', '9390.0', 'Other']


fraction_column_list= ['ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease','ChronicCond_Cancer',
'ChronicCond_ObstrPulmonary','ChronicCond_Depression','ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis',
'ChronicCond_rheumatoidarthritis','ChronicCond_stroke', 'RenalDiseaseIndicator', 'Deceased', 'Gender', 'Race', 'In_Out','PotentialFraud']

agg_steps = [('claim_period_transform', DateTransform(start='ClaimStartDt', end='ClaimEndDt', newColumn='ClaimPeriod')), 
         ('hospital_period_transform', DateTransform(start='AdmissionDt', end='DischargeDt', newColumn='HospitalDays')),
         ('age_transform', AgeTransform(dob='DOB', dod='DOD', ageColumn='Age', deceasedColumn='Deceased')),
         ('diag_code_count', CodeCountTransform(colunmsToCount = diagnosis_code_columns, newColumn='DiagCodeCounts')),
         ('proc_code_count', CodeCountTransform(colunmsToCount = procedure_code_columns, newColumn='ProcCodeCounts')),
         ('diag_code_frequency_group', CodeFrequencyGroupTransform(code_columns=diagnosis_code_columns, new_columns_prefix='ClmDiag',
                                                                  high=10000, medium_high=5000, medium=800, low=500)),
         ('proc_code_frequency_group', CodeFrequencyGroupTransform(code_columns=procedure_code_columns, new_columns_prefix='ClmProc',
                                                                  high=500, medium_high=100, medium=10, low=5)),
         ('diag_code_top15_onehot', Top15OneHotTransform(column_list=diagnosis_code_columns, top_15_codes=train_top_15_diag_codes,
                                                         new_column_prefix='DiagCode_')),
         ('proc_code_top15_onehot', Top15OneHotTransform(column_list=procedure_code_columns, top_15_codes=train_top_15_proc_codes,
                                                         new_column_prefix='ProcCode_')),
         ('aggregation', ProviderLevelAggregateTransform(fraction_column_list=fraction_column_list))]

agg_pipe = Pipeline(agg_steps)
agg_output = agg_pipe.fit_transform(merge_provider_df)
agg_output=agg_output.rename(columns={"PotentialFraud_Frac":'PotentialFraud'})
agg_output.replace([np.inf, -np.inf], 0, inplace=True)
agg_output.fillna(0,inplace=True)
# print(agg_output)
agg_output.to_csv('data/provider_level_raw.csv',index=False)

y = agg_output['PotentialFraud']
X = agg_output.drop(columns=['PotentialFraud'])
X_train_agg, X_test_agg, y_train_agg, y_test_agg = train_test_split(X, y, random_state=10, shuffle=True, test_size=0.2)

X_train_agg.to_csv('data/X_train.csv', index=False)
X_test_agg.to_csv('data/X_test.csv', index=False)
y_train_agg.to_csv('data/y_train.csv', index=False)
y_test_agg.to_csv('data/y_test.csv', index=False)

model = Fraud_Detector()
model.train()
with open('models/model.pkl', 'wb') as f:
	pickle.dump(model,f)