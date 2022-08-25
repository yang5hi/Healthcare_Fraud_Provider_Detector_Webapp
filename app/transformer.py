import pandas as pd
from datetime import date, datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline



######################################################################
###------Load and Merge------#########################################
######################################################################

# # load the data from csv to pandas dataframe
# provider = pd.read_csv("data/Train-1542865627584.csv")
# beneficiary = pd.read_csv("data/Train_Beneficiarydata-1542865627584.csv")
# inpatient = pd.read_csv("data/Train_Inpatientdata-1542865627584.csv")
# outpatient = pd.read_csv("data/Train_Outpatientdata-1542865627584.csv")

# # union/concat the inpatient and outpatient data
# concat_df=pd.concat([inpatient, outpatient],axis=0)
# merge_bene_df=concat_df.merge(beneficiary, on='BeneID', how='left')
# merge_provider_df=merge_bene_df.merge(provider, on = 'Provider', how ='left')


# diagnosis_code_columns = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 
#            'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10' ]
# procedure_code_columns = ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
#        'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6']


# train_top_15_diag_codes = ['4019', '25000', '2724', 'V5869', '4011', '42731', 'V5861', '2720', '2449',
#  '4280', '53081', '41401', '496', '2859', '41400', 'Other']
# train_top_15_proc_codes = ['4019.0', '9904.0', '2724.0', '8154.0', '66.0', '3893.0', '3995.0', '4516.0',
#  '3722.0', '8151.0', '8872.0', '9671.0','4513.0','5849.0', '9390.0', 'Other']


# fraction_column_list= ['ChronicCond_Alzheimer','ChronicCond_Heartfailure','ChronicCond_KidneyDisease','ChronicCond_Cancer',
# 'ChronicCond_ObstrPulmonary','ChronicCond_Depression','ChronicCond_Diabetes','ChronicCond_IschemicHeart','ChronicCond_Osteoporasis',
# 'ChronicCond_rheumatoidarthritis','ChronicCond_stroke', 'RenalDiseaseIndicator', 'Deceased', 'Gender', 'Race']


#######################################################################################
####################------Transformer Classes------####################################
#######################################################################################

class DateTransform(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self, start, end, newColumn):
        # save the features list internally in the class
        self.start = start
        self.end = end
        self.newColumn = newColumn
    
    def fit(self, X, y=None):
        return self
      
    def transform(self, X, y=None):
        X[self.newColumn] = self.convertDateToPeriod(X, self.start, self.end)
        X[self.newColumn] = X[self.newColumn].fillna(0)
        return X
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)    
    
    def convertDateToPeriod(self, df, startDate, endDate):
        return (pd.to_datetime(df[endDate]) - pd.to_datetime(df[startDate])).dt.days + 1


class AgeTransform(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self, dob, dod, ageColumn, deceasedColumn):
        # save the features list internally in the class
        self.dob = dob
        self.dod = dod
        self.ageColumn = ageColumn
        self.deceasedColumn = deceasedColumn
    
    def fit(self, X, y=None):
        return self
      
    def transform(self, X, y=None):
        X[self.ageColumn] = X.apply(lambda x: self.calculateAge(dob = x[self.dob], dod = x[self.dod], calulationDate = '2009-12-01'), axis = 1)
        X[self.deceasedColumn] = X[self.dod].apply(lambda x : 0 if pd.isna(x) else 1)
        return X
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)    
    
    def calculateAge(self, dob, dod, calulationDate):
        born = datetime.strptime(dob, "%Y-%m-%d").date()
        if not pd.isna(dod):
            calulationDate = datetime.strptime(dod, "%Y-%m-%d").date()
        else:
            calulationDate = datetime.strptime(calulationDate, "%Y-%m-%d").date()
        return calulationDate.year - born.year - ((calulationDate.month, calulationDate.day) < (born.month, born.day))


class CodeCountTransform(BaseEstimator, TransformerMixin):
    # initializer 
    def __init__(self, colunmsToCount, newColumn):
        # save the features list internally in the class
        self.colunmsToCount = colunmsToCount
        self.newColumn = newColumn
        
    def fit(self, X, y=None):
        return self
      
    def transform(self, X, y=None):
        X[self.newColumn] = self.countCodeNumber(X, self.colunmsToCount)
        return X
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)    
    
    def countCodeNumber(self, df, colunmsToCount):
        df_codes = df.loc[:, colunmsToCount]
        codecount = df_codes.notnull().sum(axis=1)
        return codecount


class CodeFrequencyGroupTransform(BaseEstimator, TransformerMixin):
    '''code_columns: the column list containing the claim codes
       new_column: list of new_columns
       '''
    def __init__(self, code_columns, new_columns_prefix, high, medium_high, medium, low):
        # save the features list internally in the class
        self.code_columns = code_columns
        self.new_columns_prefix = new_columns_prefix
        self.high = high
        self.medium_high = medium_high
        self.medium = medium
        self.low = low
    
    def fit(self, X, y=None):
        self.frequency_counts = self.getTotalCodeCounts(X, self.code_columns)
        self.frequency_groups = self.getFrequencyGroups(self.frequency_counts, self.high, self.medium_high, 
                                                        self.medium, self.low)
        
    def transform(self, X, y=None):
        X[self.new_columns_prefix+'HighFreqCount'] = self.codeForfrequencyGroupCounts(X, self.code_columns, self.frequency_groups[0])
        X[self.new_columns_prefix+'MediumHighFreqCount'] = self.codeForfrequencyGroupCounts(X, self.code_columns, self.frequency_groups[1])
        X[self.new_columns_prefix+'MediumFreqCount'] = self.codeForfrequencyGroupCounts(X, self.code_columns, self.frequency_groups[2])
        X[self.new_columns_prefix+'LowFreqCount'] = self.codeForfrequencyGroupCounts(X, self.code_columns, self.frequency_groups[3])
        X[self.new_columns_prefix+'RareFreqCount'] = self.codeForfrequencyGroupCounts(X, self.code_columns, self.frequency_groups[4])
        return X
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)    
    
    def mergeDictionaryWithUpdate(self, dict_1, dict_2):
        for key in dict_2:
            value = dict_2[key]
       
            if key not in dict_1:
                dict_1[key] = value
            else:
                old_value = int(dict_1[key])
                dict_1[key] = old_value + value
        return dict_1    
    
    '''Get total counts of each code'''
    def getTotalCodeCounts(self, df, columns):
        code_counts = {}
        for column in columns:
            value_counts = df[column].value_counts().to_dict()
            code_counts = self.mergeDictionaryWithUpdate(code_counts, value_counts)
        sorted_counts = dict(sorted(code_counts.items(), key=lambda item: item[1], reverse=True))
        return sorted_counts  
    
    def getFrequencyGroups(self, dictionary, high, medium_high, medium, low):
        high_frequency = []
        medium_high_frequency = []
        medium_frequency = []
        low_fequency = []
        rare_frequency = []
        for key in dictionary:
            value = dictionary[key]
            if value >= high:
                high_frequency.append(key)
            elif value < high and value >= medium_high:
                medium_high_frequency.append(key)
            elif value < medium_high and value >= medium:
                medium_frequency.append(key)
            elif value < medium and value >= low:
                low_fequency.append(key)
            else:
                rare_frequency.append(key)
        results = []
        results.extend((high_frequency, medium_high_frequency, medium_frequency, low_fequency, rare_frequency))
        return results

    def codeForfrequencyGroupCounts(self, df, columns, frequency_group):
        df_codes = df.loc[:, columns]
        codecount = df_codes.isin(frequency_group).sum(axis=1)
        return codecount

class Top15OneHotTransform(BaseEstimator, TransformerMixin):
    def __init__(self, column_list, top_15_codes, new_column_prefix):
        self.column_list = column_list
        self.top_15_codes = top_15_codes
        self.new_column_prefix = new_column_prefix
        self.codes_df = pd.DataFrame()
    
    def fit(self, X, y=None):
        #change codes not in top15 with 'Other'
        codes_df = X[self.column_list]
        for column in self.column_list:
            codes_df[column] =  codes_df[column].apply(lambda x : 'Other' if x not in self.top_15_codes else x )
        self.codes_df = codes_df    
        
    def transform(self, X, y=None):
        for code in self.top_15_codes:
            column_name = self.new_column_prefix + code
            X[column_name] = self.codeForNHotCounts(self.codes_df, self.column_list, code)
        return X
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)  
    
    def codeForNHotCounts(self, df, columns, code):
        df_codes = df.loc[:, columns]
        codecount = (df_codes==code).sum(axis=1)
        return codecount

class ProviderLevelAggregateTransform(BaseEstimator, TransformerMixin):
    def __init__(self, fraction_column_list):
        self.fraction_column_list = fraction_column_list
    
    def fit(self, X, y=None):
        #change codes not in top15 with 'Other'
        return self 
        
    def transform(self, X, y=None):
        agg_df = pd.DataFrame()
        
        agg_df = X.groupby('Provider').agg(     MedianAge = ('Age', 'median'),
                                                MeanInscClaimAmtReimbursed = ('InscClaimAmtReimbursed', 'mean'),
                                                MaxInscClaimAmtReimbursed = ('InscClaimAmtReimbursed', 'max'),
                                                TotalInscClaimAmtReimbursed = ('InscClaimAmtReimbursed', 'sum'),
                                                MeanDeductibleAmtPaid = ('DeductibleAmtPaid', 'mean'),
                                                MaxDeductibleAmtPaid = ('DeductibleAmtPaid', 'max'),
                                                MeanNumMonth_PartACov = ('NoOfMonths_PartACov','mean'),
                                                MeanNumMonth_PartBCov = ('NoOfMonths_PartBCov','mean'),
                                                MeanOPAnnualReimbursementAmt = ('OPAnnualReimbursementAmt', 'mean'),
                                                MaxOPAnnualReimbursementAmt = ('OPAnnualReimbursementAmt', 'max'),
                                                TotalOPAnnualReimbursementAmt = ('OPAnnualReimbursementAmt', 'sum'),
                                                MeanOPAnnualDeductibleAmt = ('OPAnnualDeductibleAmt', 'mean'),
                                                MaxOPAnnualDeductibleAmt = ('OPAnnualDeductibleAmt', 'max'),
                                                TotalOPAnnualDeductibleAmt = ('OPAnnualDeductibleAmt', 'sum'), 
                                                MeanIPAnnualReimbursementAmt = ('IPAnnualReimbursementAmt', 'mean'),
                                                MaxIPAnnualReimbursementAmt = ('IPAnnualReimbursementAmt', 'max'),
                                                TotalIPAnnualReimbursementAmt = ('IPAnnualReimbursementAmt', 'sum'),
                                                MeanIPAnnualDeductibleAmt = ('IPAnnualDeductibleAmt', 'mean'),
                                                MaxIPAnnualDeductibleAmt = ('IPAnnualDeductibleAmt', 'max'),
                                                TotalIPAnnualDeductibleAmt = ('IPAnnualDeductibleAmt', 'sum'),
                                                MeanClaimPeriods = ('ClaimPeriod', 'mean'),
                                                MaxHospitalDays = ('HospitalDays', 'max'), 
                                                MedianHospitalDays = ('HospitalDays', 'median'),
                                                MeanHospitalDays = ('HospitalDays', 'mean'),
                                                MaxDiagCodeNumPerClaim = ('DiagCodeCounts', 'max'),
                                                MeanDiagCodeNumPerClaim = ('DiagCodeCounts', 'mean'),
                                                MaxProcCodeNumPerClaim = ('ProcCodeCounts', 'max'),
                                                MeanProcCodeNumPerClaim = ('ProcCodeCounts', 'mean'),
                                                TotalDiagCodeNum = ('DiagCodeCounts', 'sum'),
                                                TotalProcCodeNum = ('ProcCodeCounts', 'sum'),
                                                MeanHighFreqDiagCodeNumPerClaim = ('ClmDiagHighFreqCount', 'mean'),
                                                MeanMediumHighFreqDiagCodeNumPerClaim = ('ClmDiagMediumHighFreqCount', 'mean'),
                                                MeanMediumFreqDiagCodeNumPerClaim = ('ClmDiagMediumFreqCount', 'mean'),
                                                MeanLowFreqDiagCodeNumPerClaim = ('ClmDiagLowFreqCount', 'mean'),
                                                MeanRareFreqDiagCodeNumPerClaim = ('ClmDiagRareFreqCount', 'mean'),
                                                TotalHighFreqProcCodeNumPerClaim = ('ClmProcHighFreqCount', 'sum'),
                                                TotalMediumHighFreqProcCodeNumPerClaim = ('ClmProcMediumHighFreqCount', 'sum'),
                                                TotalMediumFreqProcCodeNumPerClaim = ('ClmProcMediumFreqCount', 'sum'),
                                                TotalLowFreqProcCodeNumPerClaim = ('ClmProcLowFreqCount', 'sum'),
                                                TotalRareFreqProcCodeNumPerClaim = ('ClmProcRareFreqCount', 'sum'),
                                                totalDiagCode_4019 = ('DiagCode_4019', 'sum'),      
                                                totalDiagCode_25000 = ('DiagCode_25000','sum'),
                                                totalDiagCode_2724 = ('DiagCode_2724', 'sum'),
                                                totalDiagCode_V5869 = ('DiagCode_V5869', 'sum'),
                                                totalDiagCode_4011 = ('DiagCode_4011', 'sum'),
                                                totalDiagCode_42731 =  ('DiagCode_42731', 'sum'),
                                                totalDiagCode_V5861 = ('DiagCode_V5861', 'sum'),
                                                totalDiagCode_2720 = ('DiagCode_2720', 'sum'),
                                                totalDiagCode_2449 = ('DiagCode_2449', 'sum'),
                                                totalDiagCode_4280 = ('DiagCode_4280', 'sum'),
                                                totalDiagCode_53081 = ('DiagCode_53081', 'sum'),
                                                totalDiagCode_41401 = ('DiagCode_41401', 'sum'),
                                                totalDiagCode_496 = ('DiagCode_496', 'sum'),
                                                totalDiagCode_2589 = ('DiagCode_2859', 'sum'),
                                                totalDiagCode_41400 = ('DiagCode_41400', 'sum'),
                                                totalDiagCode_Other = ('DiagCode_Other', 'sum'),
                                                totalProcCode_4019 = ('ProcCode_4019.0', 'sum'),
                                                totalProcCode_9904 = ('ProcCode_9904.0', 'sum'),
                                                totalProcCode_2724 = ('ProcCode_2724.0', 'sum'),
                                                totalProcCode_8154 = ('ProcCode_8154.0', 'sum'),
                                                totalProcCode_66 = ('ProcCode_66.0', 'sum'),
                                                totalProcCode_3893 = ('ProcCode_3893.0', 'sum'),
                                                totalProcCode_3995 = ('ProcCode_3995.0', 'sum'),
                                                totalProcCode_4516 = ('ProcCode_4516.0', 'sum'),
                                                totalProcCode_3722 = ('ProcCode_3722.0', 'sum'),
                                                totalProcCode_8151 = ('ProcCode_8151.0', 'sum'),
                                                totalProcCode_8872 = ('ProcCode_8872.0', 'sum'),
                                                totalProcCode_9671 = ('ProcCode_9671.0', 'sum'),
                                                totalProcCode_4513 = ('ProcCode_4513.0', 'sum'),
                                                totalProcCode_5849 = ('ProcCode_5849.0', 'sum'),
                                                totalProcCode_9390 = ('ProcCode_9390.0', 'sum'),
                                                totalProcCode_Other = ('ProcCode_Other', 'sum'))
        # Caculate aggregted fraction 
        for column in self.fraction_column_list:
            new_colunm = column + '_Frac'
            agg_df[new_colunm] = (X.groupby('Provider').apply(lambda x: (x[column] == 1).sum()/x[column].count())).values                             
        
        # Total Claims per provider, unique benes per providers, claim/unique benes ratio
        agg_df['ClaimNumbers'] = (X.groupby('Provider')[['ClaimID']].count()).values
        agg_df['UniqBeneCount'] = (X.groupby('Provider')[['BeneID']].nunique()).values
        agg_df['ClaimCountsperPatient'] = agg_df['ClaimNumbers'] / agg_df['UniqBeneCount']
        
        agg_df['UniqATPhysCount'] = (X.groupby('Provider')[['AttendingPhysician']].nunique()).values
        agg_df['ClmsperATPhysn'] = agg_df['ClaimNumbers'] / agg_df['UniqATPhysCount']
        agg_df['UniqOPPhysCount'] = (X.groupby('Provider')[['OperatingPhysician']].nunique()).values
        agg_df['ClmsperOPPhysn'] = agg_df['ClaimNumbers'] / agg_df['UniqOPPhysCount']
        agg_df['UniqOTPhysCount'] = (X.groupby('Provider')[['OtherPhysician']].nunique()).values
        agg_df['ClmsperOTPhysn'] = agg_df['ClaimNumbers'] / agg_df['UniqOTPhysCount']
        
        # State, county, major race count
        agg_df['UniqStateCount'] = (X.groupby('Provider')[['State']].nunique()).values
        agg_df['UniqCountyCount'] = (X.groupby('Provider')[['County']].nunique()).values
        agg_df = agg_df.merge(self.calculateMajor(X, 'State'), on='Provider', how='left')
        agg_df = agg_df.merge(self.calculateMajor(X, 'County'), on='Provider', how='left')
        agg_df = agg_df.merge(self.calculateMajor(X, 'Race'), on='Provider', how='left')                                           
        return agg_df
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)  
    
    def calculateMajor(self, df, column):
        major_df= pd.DataFrame(df.groupby(['Provider'])[column].apply(lambda x: x.value_counts().head(1)).reset_index())
        major_df.drop(columns=[column], inplace = True)
        newMajorColumn = 'Major' + column
        major_df.rename(columns={"level_1": newMajorColumn}, inplace=True)
        return major_df


#######################################################################################
####################------All Steps Pipelines------####################################
#######################################################################################




# agg_steps = [('claim_period_transform', DateTransform(start='ClaimStartDt', end='ClaimEndDt', newColumn='ClaimPeriod')), 
#          ('hospital_period_transform', DateTransform(start='AdmissionDt', end='DischargeDt', newColumn='HospitalDays')),
#          ('age_transform', AgeTransform(dob='DOB', dod='DOD', ageColumn='Age', deceasedColumn='Deceased')),
#          ('diag_code_count', CodeCountTransform(colunmsToCount = diagnosis_code_columns, newColumn='DiagCodeCounts')),
#          ('proc_code_count', CodeCountTransform(colunmsToCount = procedure_code_columns, newColumn='ProcCodeCounts')),
#          ('diag_code_frequency_group', CodeFrequencyGroupTransform(code_columns=diagnosis_code_columns, new_columns_prefix='ClmDiag',
#                                                                   high=10000, medium_high=5000, medium=800, low=500)),
#          ('proc_code_frequency_group', CodeFrequencyGroupTransform(code_columns=procedure_code_columns, new_columns_prefix='ClmProc',
#                                                                   high=500, medium_high=100, medium=10, low=5)),
#          ('diag_code_top15_onehot', Top15OneHotTransform(column_list=diagnosis_code_columns, top_15_codes=train_top_15_diag_codes,
#                                                          new_column_prefix='DiagCode_')),
#          ('proc_code_top15_onehot', Top15OneHotTransform(column_list=procedure_code_columns, top_15_codes=train_top_15_proc_codes,
#                                                          new_column_prefix='ProcCode_')),
#          ('aggregation', ProviderLevelAggregateTransform(fraction_column_list=fraction_column_list))]

# agg_pipe = Pipeline(agg_steps)
# agg_output = agg_pipe.fit_transform(merge_provider_df)

# print(agg_output)