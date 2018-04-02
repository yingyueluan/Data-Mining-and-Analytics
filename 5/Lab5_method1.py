import pandas as pd
import numpy as np

# processing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2, f_classif
from sklearn.cluster import KMeans

# models
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier

# gridsearch and pipelining
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

##==========================================##
##         Processing Dataset               ##
##==========================================##

##=====================##
## Cleaning Data       ##
##=====================##
col_Names=["ID", "Date", "Element", "DataValue", "Mflag", "Qflag", "Sflag", "ObsTime"]
data_2017 = pd.read_csv('2017.csv', header = None, names = col_Names)
data_2016 = pd.read_csv('2016.csv', header = None, names = col_Names)
data_2015 = pd.read_csv('2015.csv', header = None, names = col_Names)
data_2014 = pd.read_csv('2014.csv', header = None, names = col_Names)

elements = data_2017['Element'].unique()
counts = {}
for e in elements:
     counts[e] = len(data_2017[data_2017.Element == e])
     
elms = sorted(counts, key=counts.get, reverse = True)[:15]
df = data_2017[data_2017['Element'].isin(elms)]

## Combine columns ID and Date into one column
df['Sub_ID'] = df['Date'].astype(str) + df['ID'] 

df = df.drop(["Mflag", "Qflag", "Sflag", "ObsTime"], axis = 1)
df.head()

df_T = df.pivot(index='Sub_ID', columns='Element', values='DataValue').reset_index()
df_T.head()

df_T["Month"] = df_T["Sub_ID"].str[4:6]
df_T["Day"] = df_T["Sub_ID"].str[6:8]
df_T.head()

# df_T.to_csv("clean_2017.csv")
# repeat process to get clean data for 2014, 2015 and 2016

##=====================##
## Get first 2 month   ##
##=====================##
def get_data12(file):
    dat = pd.read_csv(file)
    dat_12 = dat[dat.Month.isin([1, 2])]
    return dat_12

dat2014_12 = get_data12("clean_2014.csv")
dat2015_12 = get_data12("clean_2015.csv")
dat2016_12 = get_data12("clean_2016.csv")
dat2017_12 = get_data12("clean_2017.csv")

##=====================##
## Get training set    ##
##=====================##
def pick_features(data):
    data = data[['Sub_ID', 'Month', 'Day', 'TAVG', 'TMAX', 'TMIN']]
    return data

dat2014_12 = pick_features(dat2014_12)
dat2015_12 = pick_features(dat2015_12)
dat2016_12 = pick_features(dat2016_12)
dat2017_12 = pick_features(dat2017_12)

##=========================##
## Get test set ##
##=========================##
dat2017_12["ID"] = dat2017_12["Sub_ID"].str[8:]
dat_part, dat_test = train_test_split(dat2017_12, test_size=0.5)

# dat_test.to_csv("dat_test12_2.csv", index = False)

dat_train = pd.concat([dat2014_12,dat2015_12,dat2016_12,dat_part], ignore_index=True)
dat_train["ID"] = dat_train["Sub_ID"].str[8:]

# dat_train.to_csv("dat_train12_2.csv", index = False)

##===========================##
## Get values for prediction ##
##===========================##
dat_pred = dat_train.groupby(['ID','Month','Day']).mean().reset_index()
dat_pred = dat_pred.drop(['TMIN'], axis = 1)

sample_submission = pd.read_csv("sample_submission.csv")
sample_submission["Month"] = sample_submission["SUB_ID"].str[4:6].astype(str).astype(int)
sample_submission["Day"] = sample_submission["SUB_ID"].str[6:8].astype(str).astype(int)
sample_submission["ID"] = sample_submission["SUB_ID"].str[8:]

dat_pred = pd.merge(sample_submission, dat_pred, how = 'left', on=['ID', 'Month', 'Day'])

# dat_pred.to_csv("dat_pred_2.csv", index = False)

##==========================================##
##      Fitting Model on training set       ##
##==========================================##
def fill_missing(dat):
    # dat[['TAVG','TMAX']] = dat[['TAVG','TMAX']].fillna(-9999)
    group_dat = dat_train.groupby(['Month']).mean().reset_index()
    min1 = group_dat['TMIN'][0]
    min2 = group_dat['TMIN'][1]
    avg1 = group_dat['TAVG'][0]
    avg2 = group_dat['TAVG'][1]
    max1 = group_dat['TMAX'][0]
    max2 = group_dat['TMAX'][1]
    
    dat1 = dat[dat.Month == 1]
    dat1['TMIN'] = dat1['TMIN'].fillna(min1)
    dat1['TAVG'] = dat1['TAVG'].fillna(avg1)
    dat1['TMAX'] = dat1['TMAX'].fillna(max1)
    dat2 = dat[dat.Month == 2]
    dat2['TMIN'] = dat2['TMIN'].fillna(min2)
    dat2['TAVG'] = dat2['TAVG'].fillna(avg2)
    dat2['TMAX'] = dat2['TMAX'].fillna(max2)
    
    dat = pd.concat([dat1, dat2], ignore_index=True)
    return dat

def fill_missing_pred(dat):
    # dat[['TAVG','TMAX']] = dat[['TAVG','TMAX']].fillna(-9999)
    group_dat = dat_train.groupby(['Month']).mean().reset_index()

    avg1 = group_dat['TAVG'][0]
    avg2 = group_dat['TAVG'][1]
    max1 = group_dat['TMAX'][0]
    max2 = group_dat['TMAX'][1]
    
    dat1 = dat[dat.Month == 1]

    dat1['TAVG'] = dat1['TAVG'].fillna(avg1)
    dat1['TMAX'] = dat1['TMAX'].fillna(max1)
    dat2 = dat[dat.Month == 2]

    dat2['TAVG'] = dat2['TAVG'].fillna(avg2)
    dat2['TMAX'] = dat2['TMAX'].fillna(max2)
    
    dat = pd.concat([dat1, dat2], ignore_index=True)
    return dat
##========================##
## Filling missing values ##
##========================##
dat_train = fill_missing(dat_train)
#==============================================================================
# dat_train['TAVG'] = dat_train['TAVG'].fillna(150)
# dat_train['TMAX'] = dat_train['TMAX'].fillna(180)
# 
# group_dat = dat_train.groupby(['Month']).mean().reset_index()
# avg1 = group_dat['TMIN'][0]
# avg2 = group_dat['TMIN'][1]
# 
# dat_train1 = dat_train[dat_train.Month == 1]
# dat_train1['TMIN'] = dat_train1['TMIN'].fillna(avg1)
# dat_train2 = dat_train[dat_train.Month == 2]
# dat_train2['TMIN'] = dat_train2['TMIN'].fillna(avg2)
# 
# dat_train = pd.concat([dat_train1, dat_train2], ignore_index=True)
#==============================================================================

##==============================##
## Factorize/normalize features ##
##==============================##
target = dat_train["TMIN"]

feature_dic = dat_train[["Month", "Day"]].to_dict('records')
vec = DictVectorizer()
feature = vec.fit_transform(feature_dic).toarray()
f_train = pd.DataFrame(feature)
f_train = f_train.rename(columns={0: "Day", 1: "Month"})

TAVG = dat_train[['TAVG']].values
TAVG_scaled = StandardScaler().fit_transform(TAVG)

TMAX = dat_train[['TMAX']].values
TMAX_scaled = StandardScaler().fit_transform(TMAX)

f_train['TAVG'] = TAVG_scaled
f_train['TMAX'] = TMAX_scaled

##========================##
## Fitting Model          ##
##========================##

# Stochastic Gradient Descent
clf = SGDRegressor()
clf.fit(f_train, target)

# Gradient Boosting
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=3, random_state=0, loss='huber')
est.fit(f_train, target)

# AdaBoost
ada = AdaBoostRegressor()
ada.fit(f_train, target)

# Random Forest
rf = RandomForestRegressor(max_features=2, min_samples_split=4, n_estimators=50, min_samples_leaf=2)
rf.fit(f_train, target)

# Ridge
ridge = Ridge(alpha=1.0, solver = 'auto')
ridge.fit(f_train, target)

# Voting
estimators = []
model1 = SGDRegressor()
estimators.append(('SGDR', model1))
model2 = Ridge()
estimators.append(('Ridge', model2))
eclf = VotingClassifier(estimators)
eclf.fit(f_train, target)
#==============================================================================
# # Naive Bayes
# mnb = MultinomialNB()
# mnb.fit(f_train, target)
# 
#==============================================================================

##==========================================##
##    Evaluating Model on test set    ##
##==========================================##

##========================##
## Filling missing values ##
##========================##
dat_test = fill_missing(dat_test)
#==============================================================================
# dat_test['TAVG'] = dat_test['TAVG'].fillna(115)
# dat_test['TMAX'] = dat_test['TMAX'].fillna(110)
# 
# group_dat = dat_test.groupby(['Month']).mean().reset_index()
# avg1 = group_dat['TMIN'][0]
# avg2 = group_dat['TMIN'][1]
# 
# dat_test1 = dat_test[dat_test.Month == 1]
# dat_test1['TMIN'] = dat_test1['TMIN'].fillna(avg1)
# dat_test2 = dat_test[dat_test.Month == 2]
# dat_test2['TMIN'] = dat_test2['TMIN'].fillna(avg2)
# 
# dat_test = pd.concat([dat_test1, dat_test2], ignore_index=True)
#==============================================================================

##==============================##
## Factorize/normalize features ##
##==============================##
target_test = dat_test["TMIN"]

feature_dic = dat_test[["Month", "Day"]].to_dict('records')
vec = DictVectorizer()
feature = vec.fit_transform(feature_dic).toarray()
f_test = pd.DataFrame(feature)
f_test = f_test.rename(columns={0: "Day", 1: "Month"})

TAVG = dat_test[['TAVG']].values
TAVG_scaled = StandardScaler().fit_transform(TAVG)

TMAX = dat_test[['TMAX']].values
TMAX_scaled = StandardScaler().fit_transform(TMAX)

f_test['TAVG'] = TAVG_scaled
f_test['TMAX'] = TMAX_scaled

##========================##
## Get accuracy Score     ##
##========================##
target_pred = ridge.predict(f_test)
score = mean_absolute_error(target_test, target_pred)
## SGDRegressor: 46.667
## GradientBoostingRegressor: 59.332



##==========================================##
##           Getting Prediction             ##
##==========================================##
##========================##
## Filling missing values ##
##========================##
dat_pred = fill_missing_pred(dat_pred)
#==============================================================================
# dat_pred['TAVG'] = dat_pred['TAVG'].fillna(-4)
# dat_pred['TMIN'] = dat_pred['TMIN'].fillna(40)
# 
#==============================================================================
##==============================##
## Factorize/normalize features ##
##==============================##
feature_dic = dat_pred[["Month", "Day"]].to_dict('records')
vec = DictVectorizer()
feature = vec.fit_transform(feature_dic).toarray()
f_pred = pd.DataFrame(feature)
f_pred = f_pred.rename(columns={0: "Day", 1: "Month"})

TAVG = dat_pred[['TAVG']].values
TAVG_scaled = StandardScaler().fit_transform(TAVG)

TMAX = dat_pred[['TMAX']].values
TMAX_scaled = StandardScaler().fit_transform(TMAX)

f_pred['TAVG'] = TAVG_scaled
f_pred['TMAX'] = TMAX_scaled

##========================##
## Get prediction set     ##
##========================##
TMIN_pred = ridge.predict(f_pred)
dat_pred['DATA_VALUE'] = TMIN_pred
dat_pred = dat_pred[['SUB_ID', 'DATA_VALUE']]

#dat_pred.to_csv("submission.csv", index = False)


#==============================================================================
# Clustering
# X = dat_train[["TAVG","TMIN","TMAX"]]
# kmeans = KMeans(n_clusters=8, random_state=0).fit(X)
# dat_train['labels'] = kmeans.labels_
#==============================================================================







