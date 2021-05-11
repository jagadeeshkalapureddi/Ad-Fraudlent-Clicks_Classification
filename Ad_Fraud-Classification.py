# # --------------------@ FRAUDLENT CLICK PREDICTION ANALYSIS @------------------------------
# # -----------------------! FULL CLASSIFICATION ANALYSIS !----------------------------------

# # IMPORT THE REQUIRED PACKAGES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import gc # for deleting unused variables
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus,graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/jagad/AppData/Local/Programs/Python/Python38-32/Lib/site-packages/graphviz/bin'


# ### READ THE DATASET

df = pd.read_csv('ad_fraud.csv')

# ### DATA UNDERSTANDING

# ## Understanding and Exploring the Data
# 
# The data contains observations of about 240 million clicks, and whether a given click resulted in a download or not (1/0). 
# 
# On Kaggle, the data is split into train.csv and train_sample.csv (100,000 observations). We'll use the smaller train_sample.csv in this notebook for speed, though while training the model for Kaggle submissions, the full training data will obviously produce better results.
# 
# The detailed data dictionary is mentioned here:
# - ```ip```: ip address of click.
# - ```app```: app id for marketing.
# - ```device```: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# - ```os```: os version id of user mobile phone
# - ```channel```: channel id of mobile ad publisher
# - ```click_time```: timestamp of click (UTC)
# - ```attributed_time```: if user download the app for after clicking an ad, this is the time of the app download
# - ```is_attributed```: the target that is to be predicted, indicating the app was downloaded
# 
# Let's try finding some useful trends in the data.

# In[96]:


# Check for the Data 
df.head(3)

df.info()

df.shape

df.columns

df.isna().sum()

# ## EXPLORATORY DATA ANALYSIS (EDA)

# Drop attribute time because it having less data (227 results out of 100000), So iam removing that attribute.
df.drop('attributed_time', axis = 1, inplace = True)

# Again check the columns for confirmation.
print('Total_Columns: ', len(df.columns))
print(df.columns)
print('Shape :',df.shape)

# ### DATA SPLITING INTO X AND Y

df.iloc[:, 0:6]

# Displays memory consumed by each column ---
print(df.memory_usage())
# space used by training data
print('Training dataset uses {0} MB'.format(df.memory_usage().sum()/1024**2))

# ### Exploring the Data - Univariate Analysis

# Let's now understand and explore the data. Let's start with understanding the size and data types of the train_sample data.

# look at non-null values, number of entries etc.
# there are no missing values

df.info()

# Basic exploratory analysis 

# Number of unique values in each column
def fraction_unique(x):
    return len(df[x].unique())

number_unique_vals = {x: fraction_unique(x) for x in df.columns}
number_unique_vals

# All columns apart from click time are originally int type, 
# though note that they are all actually categorical 
df.dtypes

# # distribution of 'app' 
# # some 'apps' have a disproportionately high number of clicks (>15k), and some are very rare (3-4)
plt.figure(figsize=(14, 8))
sns.countplot(x="app", data=df)
plt.show()

# # distribution of 'device' 
# # this is expected because a few popular devices are used heavily
plt.figure(figsize=(14, 8))
sns.countplot(x="device", data=df)
plt.show()

# # channel: various channels get clicks in comparable quantities
plt.figure(figsize=(14, 8))
sns.countplot(x="channel", data=df)
plt.show()

# # os: there are a couple commos OSes (android and ios?), though some are rare and can indicate suspicion 
plt.figure(figsize=(14, 8))
sns.countplot(x="os", data=df)
plt.show()

# Let's now look at the distribution of the target  variable 'is_attributed'.

# Check for the Shape and value counts of x and y after splitting the data variables.
y = df.iloc[:, 6]
print('Dependent_Variable: ','\n','Dimensions: ', y.shape,'\n','Column_name: ', y.name, '\n')
print('Value_counts: ', '\n',y.value_counts(normalize = True).mul(100).round(1).astype(str) + '%', '\n')
x = df.iloc[:,0:6]
print('Independent_Variable: ','\n','Dimensions: ', x.shape,'\n','Column_names: ', x.columns, '\n')


# Only **about 0.2% of clicks are 'fraudulent'**, which is expected in a fraud detection problem. Such high class imbalance is probably going to be the toughest challenge of this problem.

# ### Exploring the Data - Segmented Univariate Analysis
# 
# Let's now look at how the target variable varies with the various predictors.

# In[113]:


# plot the average of 'is_attributed', or 'download rate'
# with app (clearly this is non-readable)
app_target = df.groupby('app').is_attributed.agg(['mean', 'count'])
app_target


# This is clearly non-readable, so let's first get rid of all the apps that are very rare (say which comprise of less than 20% clicks) and plot the rest.

# In[114]:


frequent_apps = df.groupby('app').size().reset_index(name='count')
frequent_apps = frequent_apps[frequent_apps['count']>frequent_apps['count'].quantile(0.80)]
frequent_apps = frequent_apps.merge(df, on='app', how='inner')
frequent_apps.head()


# In[115]:


plt.figure(figsize=(10,10))
sns.countplot(y="app", hue="is_attributed", data=frequent_apps);


# You can do lots of other interesting ananlysis with the existing features. For now, let's create some new features which will probably improve the model.

# ## Feature Engineering

# Let's now derive some new features from the existing ones. There are a number of features one can extract from ```click_time``` itself, and by grouping combinations of IP with other features.

# ### Datetime Based Features
# 

# In[116]:


# Creating datetime variables
# takes in a df, adds date/time based columns to it, and returns the modified df
def timeFeatures(df):
    # Derive new features using the click_time column
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    return df


# In[117]:


# creating new datetime variables and dropping the old ones
df = timeFeatures(df)
df.drop(['click_time', 'datetime'], axis=1, inplace=True)
df.head()


# In[118]:


df.info()


# In[119]:


# datatypes
# note that by default the new datetime variables are int64
df.dtypes


# In[120]:


# memory used by training data
print('Training dataset uses {0} MB'.format(df.memory_usage().sum()/1024**2))


# In[121]:


# lets convert the variables back to lower dtype again
int_vars = ['app', 'device', 'os', 'channel', 'day_of_week','day_of_year', 'month', 'hour']
df[int_vars] = df[int_vars].astype('uint16')


# In[122]:


df.dtypes


# In[123]:


# space used by training data
print('Training dataset uses {0} MB'.format(df.memory_usage().sum()/1024**2))


# In[124]:


# garbage collect (unused) object
gc.collect()


# ## Modelling
# 
# Let's now build models to predict the variable ```is_attributed``` (downloaded). We'll try the several variants of boosting (adaboost, gradient boosting and XGBoost), tune the hyperparameters in each model and choose the one which gives the best performance.
# 
# In the original Kaggle competition, the metric for model evaluation is **area under the ROC curve**.
# 

# In[196]:


# create x and y train
x = df.drop('is_attributed', axis=1)
y = df[['is_attributed']]

# split data into train and test/validation sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[197]:


# check the average download rates in train and test data, should be comparable
print(y_train.mean())
print(y_test.mean())


# ## Logistic Regression
# ### Model Evaluation :
# 
# ### Full Model:

# ### VIF 

# In[198]:


vif_data = pd.DataFrame() 
vif_data["feature"] = df.columns

vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                          for i in range(len(df.columns))] 
  
print(vif_data)


# ### LOGIT OLS SUMMARY

# In[199]:


x_train_sm = sm.Logit(y_train, x_train)
lm = x_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm.summary())


# ### Fit the train model and predict the test set

# In[200]:


glm = LogisticRegression()
glm = glm.fit(x_train, y_train)
predicted = glm.predict(x_test)


# ### RFE

# In[201]:


rfe = RFE(glm, 10)
rfe = rfe.fit(x_train, y_train)
print(rfe.support_)
print(rfe.ranking_)


# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# In[202]:


# Function For Logistic Regression Create Summary For Logistic Regression

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def get_summary(y_test,predicted):
    # Confusion Matrix
    conf_mat = confusion_matrix(y_test,predicted)
    TP = conf_mat[0,0:1]
    FP = conf_mat[0,1:2]
    FN = conf_mat[1,0:1]
    TN = conf_mat[1,1:2]
    
    accuracy = (TP+TN)/((FN+FP)+(TP+TN))
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall =  TP / (TP + FN)
    fScore = (2 * recall * precision) / (recall + precision)
    auc = roc_auc_score(y_test, predicted)

    print("Confusion Matrix:\n",conf_mat)
    print("Accuracy:",accuracy)
    print("Sensitivity :",sensitivity)
    print("Specificity :",specificity)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-score:",fScore)
    print("AUC:",auc)
    print('\n')
    print("ROC curve:")
    fpr, tpr, thresholds = roc_curve(y_test, predicted)
    plot_roc_curve(fpr, tpr)


# In[203]:


get_summary(y_test,predicted)


# ### Model - 2

# In[204]:


x1_train = x_train[['app', 'device', 'os', 'channel', 'day_of_week', 'day_of_year', 'month', 'hour']]
x1_test = x_test[['app', 'device', 'os', 'channel', 'day_of_week', 'day_of_year', 'month', 'hour']]


# ### LOGIT OLS SUMMARY

# In[205]:


x1_train_sm = sm.Logit(y_train, x1_train)
lm1 = x1_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm1.summary())


# ### Fit the train model and predict the test set

# In[206]:


glm1 = LogisticRegression()
glm1 = glm1.fit(x1_train, y_train)
predicted1 = glm1.predict(x1_test)


# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# In[207]:


# Function For Logistic Regression Create Summary For Logistic Regression

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def get_summary(y_test,predicted1):
    # Confusion Matrix
    conf_mat = confusion_matrix(y_test,predicted1)
    TP = conf_mat[0,0:1]
    FP = conf_mat[0,1:2]
    FN = conf_mat[1,0:1]
    TN = conf_mat[1,1:2]
    
    accuracy = (TP+TN)/((FN+FP)+(TP+TN))
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall =  TP / (TP + FN)
    fScore = (2 * recall * precision) / (recall + precision)
    auc = roc_auc_score(y_test, predicted1)

    print("Confusion Matrix:\n",conf_mat)
    print("Accuracy:",accuracy)
    print("Sensitivity :",sensitivity)
    print("Specificity :",specificity)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-score:",fScore)
    print("AUC:",auc)
    print('\n')
    print("ROC curve:")
    fpr, tpr, thresholds = roc_curve(y_test, predicted1)
    plot_roc_curve(fpr, tpr)


# In[208]:


get_summary(y_test,predicted1)


# ### Decision Tree

# In[209]:


dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(x_train,y_train)


# In[210]:


# making predictions
y_pred_default = dt_default.predict(x_test)

# Printing classifier report after prediction
print(classification_report(y_test,y_pred_default))


# In[211]:


# Printing confusion matrix and accuracy
print(confusion_matrix(y_test,y_pred_default))
print(accuracy_score(y_test,y_pred_default))


# In[166]:


# Importing required packages for visualization
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus,graphviz

# Putting features
features = list(df.columns[[0,1,2,3,4,6,7,8,9]])
features


# In[167]:


# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(dt_default, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[168]:


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(1, 40)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score = True)
tree.fit(x_train, y_train)


# In[169]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[170]:


# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[171]:


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score = True)
tree.fit(x_train, y_train)


# In[172]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[173]:


# plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[174]:


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_split': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score = True)
tree.fit(x_train, y_train)


# In[175]:


# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()


# In[176]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ['gini', "entropy"]
}

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, verbose = 1)

# Fit the grid search to the data
grid_search.fit(x_train,y_train)


# In[177]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.head(3)


# In[178]:


# printing the optimal accuracy score and hyperparameters
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)


# In[179]:


# model with optimal hyperparameters
clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=10, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(x_train, y_train)


# In[180]:


# accuracy score
clf_gini.score(x_test,y_test)


# In[181]:


# plotting the tree
dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[182]:


# tree with max_depth = 3
clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=3, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(x_train, y_train)

# score
print(clf_gini.score(x_test,y_test))


# In[183]:


# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[184]:


y_pred = clf_gini.predict(x_test)
print(classification_report(y_test, y_pred))


# In[185]:


# confusion matrix
print(confusion_matrix(y_test,y_pred))


# ### RANDOM FOREST MODEL

# In[187]:


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)


# In[188]:


rfc_pred  = rfc.predict(x_test)
rfc_pred


# ## MODEL BUILDING

# In[189]:


print(classification_report(y_test,rfc_pred))


# ## CONFUSION MATRIX

# In[190]:


print(confusion_matrix(y_test,rfc_pred))


# ## ACCURACY

# In[191]:


rfc_accuracy = (17+1)/len(rfc_pred)
rfc_accuracy


# In[192]:


rfc_mis_cla_rate = (1+2)/len(rfc_pred)
rfc_mis_cla_rate


# In[193]:


print(classification_report(y_test,rfc_pred))


# ### KNN

# In[212]:


# This scaling process is done only on numerical values (Not include discrete variables)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_traink = sc.fit_transform(x_train)   
x_testk = sc.transform(x_test)


# In[21]:


#p : integer, optional (default = 2)
#Power parameter for the Minkowski metric. 
#When p = 1, this is equivalent to using manhattan_distance (l1), and 
#euclidean_distance (l2) for p = 2. 
#For arbitrary p, minkowski_distance (l_p) is used.


# In[213]:


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 11, metric = 'minkowski', p = 2)

classifier.fit(x_traink, y_train)

y_pred = classifier.predict(x_testk)


# In[214]:


from sklearn.metrics import accuracy_score
accuracy_score (y_test, y_pred)


# In[215]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[216]:


accuracy_list = []
error_rate = []
for i in range(1,40,2):
    knn = KNeighborsClassifier(n_neighbors=i)
#Train the model using the training sets
    knn.fit(x_traink, y_train)
#Predict the response for test dataset
    y_pred = knn.predict(x_testk)
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred), 'for' ,i)
    accuracy_list.append(metrics.accuracy_score(y_test, y_pred))
    error_rate.append(np.mean(y_pred != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40,2),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# # Predicting the test set results

# In[29]:


pred = classifier.predict(x_test)


# In[30]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[33]:


print(classification_report(y_test,pred))


# In[34]:


from sklearn import metrics


# In[35]:


f1 = metrics.f1_score(y_test,pred)
print(f1)


# In[38]:


TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]


# In[39]:


print(accuracy_score(y_test,pred))
# Alternate way forfinding accurcay
print(metrics.accuracy_score(y_test,pred))


# In[40]:


fpr, tpr, thresholds = metrics.roc_curve(y_test,pred)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.title('ROC curve classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[41]:


# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test,pred))


# # Increase the k value and check once more

# In[42]:


classifier = KNeighborsClassifier(n_neighbors = 38, metric = 'minkowski', p = 2)


# In[43]:


classifier.fit(x_train,y_train)


# In[44]:


pred = classifier.predict(x_test)


# In[47]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[48]:


print(confusion_matrix(y_test,pred))


# In[49]:


print(classification_report(y_test,pred))


# In[50]:


# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test,pred))


# In[51]:


confusion = metrics.confusion_matrix(y_test,pred)
print(confusion)


# In[52]:


f1 = metrics.f1_score(y_test,pred)
print(f1)


# In[53]:


TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]


# In[54]:


print(accuracy_score(y_test,pred))
# Alternate way forfinding accurcay
print(metrics.accuracy_score(y_test,pred))


# In[55]:


fpr, tpr, thresholds = metrics.roc_curve(y_test,pred)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.title('ROC curve classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[56]:


# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test,pred))


# # by For loop

# In[57]:


from sklearn import metrics
for i in range(1,40,2):
    classifier = KNeighborsClassifier(n_neighbors=i)
#Train the model using the training sets
    classifier.fit(x_train, y_train)
#Predict the response for test dataset
    pred = classifier.predict(x_test)
    Accuracy = []  
    print("Accuracy:",metrics.accuracy_score(y_test,pred), "for",i)


# In[58]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    Classifier = KNeighborsClassifier(n_neighbors=i)
    Classifier.fit(x_train,y_train)
    pred_i = Classifier.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[59]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:





# ### AdaBoost

# In[72]:


# adaboost classifier with max 600 decision trees of depth=2
# learning_rate/shrinkage=1.5

# base estimator
tree = DecisionTreeClassifier(max_depth=2)

# adaboost with the tree as base estimator
adaboost_model_1 = AdaBoostClassifier(
    base_estimator=tree,
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")


# In[73]:


# fit
adaboost_model_1.fit(X_train, y_train)


# In[75]:


# predictions
# the second column represents the probability of a click resulting in a download
predictions = adaboost_model_1.predict_proba(X_test)
predictions[:10]


# In[76]:


# metrics: AUC
metrics.roc_auc_score(y_test, predictions[:,1])


# ### AdaBoost - Hyperparameter Tuning
# 
# Let's now tune the hyperparameters of the AdaBoost classifier. In this case, we have two types of hyperparameters - those of the component trees (max_depth etc.) and those of the ensemble (n_estimators, learning_rate etc.). 
# 
# 
# We can tune both using the following technique - the keys of the form ```base_estimator_parameter_name``` belong to the trees (base estimator), and the rest belong to the ensemble.

# In[77]:


# parameter grid
param_grid = {"base_estimator__max_depth" : [2, 5],
              "n_estimators": [200, 400, 600]
             }


# In[78]:


# base estimator
tree = DecisionTreeClassifier()

# adaboost with the tree as base estimator
# learning rate is arbitrarily set to 0.6, we'll discuss learning_rate below
ABC = AdaBoostClassifier(
    base_estimator=tree,
    learning_rate=0.6,
    algorithm="SAMME")


# In[79]:


# run grid search
folds = 3
grid_search_ABC = GridSearchCV(ABC, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               return_train_score=True,                         
                               verbose = 1)


# In[80]:


# fit 
grid_search_ABC.fit(X_train, y_train)


# In[85]:


# cv results
cv_results = pd.DataFrame(grid_search_ABC.cv_results_)
cv_results


# In[86]:


# plotting AUC with hyperparameter combinations

plt.figure(figsize=(16,6))
for n, depth in enumerate(param_grid['base_estimator__max_depth']):
    

    # subplot 1/n
    plt.subplot(1,3, n+1)
    depth_df = cv_results[cv_results['param_base_estimator__max_depth']==depth]

    plt.plot(depth_df["param_n_estimators"], depth_df["mean_test_score"])
    plt.plot(depth_df["param_n_estimators"], depth_df["mean_train_score"])
    plt.xlabel('n_estimators')
    plt.ylabel('AUC')
    plt.title("max_depth={0}".format(depth))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')

    


# The results above show that:
# - The ensemble with max_depth=5 is clearly overfitting (training auc is almost 1, while the test score is much lower)
# - At max_depth=2, the model performs slightly better (approx 95% AUC) with a higher test score 
# 
# Thus, we should go ahead with ```max_depth=2``` and ```n_estimators=200```.
# 
# Note that we haven't experimented with many other important hyperparameters till now, such as ```learning rate```, ```subsample``` etc., and the results might be considerably improved by tuning them. We'll next experiment with these hyperparameters.

# In[87]:


# model performance on test data with chosen hyperparameters

# base estimator
tree = DecisionTreeClassifier(max_depth=2)

# adaboost with the tree as base estimator
# learning rate is arbitrarily set, we'll discuss learning_rate below
ABC = AdaBoostClassifier(
    base_estimator=tree,
    learning_rate=0.6,
    n_estimators=200,
    algorithm="SAMME")

ABC.fit(X_train, y_train)


# In[89]:


# predict on test data
predictions = ABC.predict_proba(X_test)
predictions[:10]


# In[90]:


# roc auc
metrics.roc_auc_score(y_test, predictions[:, 1])


# ### Gradient Boosting Classifier
# 
# Let's now try the gradient boosting classifier. We'll experiment with two main hyperparameters now - ```learning_rate``` (shrinkage) and ```subsample```. 
# 
# By adjusting the learning rate to less than 1, we can regularize the model. A model with higher learning_rate learns fast, but is prone to overfitting; one with a lower learning rate learns slowly, but avoids overfitting.
# 
# Also, there's a trade-off between ```learning_rate``` and ```n_estimators``` - the higher the learning rate, the lesser trees the model needs (and thus we usually tune only one of them).
# 
# Also, by subsampling (setting ```subsample``` to less than 1), we can have the individual models built on random subsamples of size ```subsample```. That way, each tree will be trained on different subsets and reduce the model's variance.

# In[91]:


# parameter grid
param_grid = {"learning_rate": [0.2, 0.6, 0.9],
              "subsample": [0.3, 0.6, 0.9]
             }


# In[92]:


# adaboost with the tree as base estimator
GBC = GradientBoostingClassifier(max_depth=2, n_estimators=200)


# In[93]:


# run grid search
folds = 3
grid_search_GBC = GridSearchCV(GBC, 
                               cv = folds,
                               param_grid=param_grid, 
                               scoring = 'roc_auc', 
                               return_train_score=True,                         
                               verbose = 1)

grid_search_GBC.fit(X_train, y_train)


# In[95]:


cv_results = pd.DataFrame(grid_search_GBC.cv_results_)
cv_results.head()


# In[96]:


# # plotting
plt.figure(figsize=(16,6))


for n, subsample in enumerate(param_grid['subsample']):
    

    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# It is clear from the plot above that the model with a lower subsample ratio performs better, while those with higher subsamples tend to overfit. 
# 
# Also, a lower learning rate results in less overfitting.

# ### XGBoost
# 
# Let's finally try XGBoost. The hyperparameters are the same, some important ones being ```subsample```, ```learning_rate```, ```max_depth``` etc.
# 

# In[99]:


conda install -c anaconda py-xgboost


# In[97]:


# fit model on training data with default hyperparameters
model = XGBClassifier()
model.fit(X_train, y_train)


# In[189]:


# make predictions for test data
# use predict_proba since we need probabilities to compute auc
y_pred = model.predict_proba(X_test)
y_pred[:10]


# In[190]:


# evaluate predictions
roc = metrics.roc_auc_score(y_test, y_pred[:, 1])
print("AUC: %.2f%%" % (roc * 100.0))


# The roc_auc in this case is about 0.95% with default hyperparameters. Let's try changing the hyperparameters - an exhaustive list of XGBoost hyperparameters is here: http://xgboost.readthedocs.io/en/latest/parameter.html
# 

# Let's now try tuning the hyperparameters using k-fold CV. We'll then use grid search CV to find the optimal values of hyperparameters.

# In[197]:


# hyperparameter tuning with XGBoost

# creating a KFold object 
folds = 3

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train, y_train)       

# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_max_depth'] = cv_results['param_max_depth'].astype('float')
cv_results.head()

# # plotting
plt.figure(figsize=(16,6))

param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 


for n, subsample in enumerate(param_grid['subsample']):
    

    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')

# The results show that a subsample size of 0.6 and learning_rate of about 0.2 seems optimal. 
# Also, XGBoost has resulted in the highest ROC AUC obtained (across various hyperparameters). 
# Let's build a final model with the chosen hyperparameters.

# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc
params = {'learning_rate': 0.2,
          'max_depth': 2, 
          'n_estimators':200,
          'subsample':0.6,
         'objective':'binary:logistic'}

# fit model on training data
model = XGBClassifier(params = params)
model.fit(X_train, y_train)

# predict
y_pred = model.predict_proba(X_test)
y_pred[:10]

# The first column in y_pred is the P(0), i.e. P(not fraud), and the second column is P(1/fraud).

# roc_auc
auc = sklearn.metrics.roc_auc_score(y_test, y_pred[:, 1])
auc

# Finally, let's also look at the feature importances.

# feature importance
importance = dict(zip(X_train.columns, model.feature_importances_))
importance

# plot
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()
