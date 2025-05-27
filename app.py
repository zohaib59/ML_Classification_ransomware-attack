import pandas as pd
import numpy as np
import seaborn as sns
import os
import pylab as pl
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv("ransomware.csv")

data.head()

pd.set_option('display.max_columns', None)

data.head()

#Check for missing values
data.isnull().sum()

#Check for duplicate values
data[data.duplicated()]


# To show Outliers in the data set run the code 

num_vars = data.select_dtypes(include=['int','float']).columns.tolist()

num_cols = len(num_vars)
num_rows = (num_cols + 2 ) // 3
fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
axs = axs.flatten()

for i, var in enumerate (num_vars):
    sns.boxplot(x=data[var],ax=axs[i])
    axs[i].set_title(var)

if num_cols < len(axs):
  for i in range(num_cols , len(axs)):
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()



def pintu (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["BitcoinAddresses"])

data = pintu(data,"BitcoinAddresses")


from autoviz.AutoViz_Class import AutoViz_Class 
AV = AutoViz_Class()
import matplotlib.pyplot as plt
%matplotlib INLINE
filename = 'ransomware.csv'
sep =","
dft = AV.AutoViz(
    filename  
)

#To convert categorical into numerics run this code

from sklearn import preprocessing
for col in data.select_dtypes(include=['object']).columns:
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(data[col].unique())
    data[col] = label_encoder.transform(data[col])
    print(f'{col} : {data[col].unique()}')



#Segregrating dataset into X and y

X = data.drop("Benign", axis = 1)

y = data["Benign"]

X.head()

y.head()

#Splitting the dataset into testing and training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Scale the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize classifiers
log_model = LogisticRegression()
rf_model = RandomForestClassifier(n_jobs=-1)
gnb_model = GaussianNB()
dt_model = DecisionTreeClassifier()

# Model fitting
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gnb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)

# Function to evaluate model performance on both training and testing data
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"--- {model_name} Evaluation ---")
    
    # Predictions for training and testing
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training performance
    print("Training Performance:")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")
    print(f"Precision: {precision_score(y_train, y_train_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_train, y_train_pred, average='weighted')}")
    print(f"F1 score: {f1_score(y_train, y_train_pred, average='weighted')}")
    print("-" * 40)
    
    # Testing performance
    print("Testing Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"Precision: {precision_score(y_test, y_test_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_test_pred, average='weighted')}")
    print(f"F1 score: {f1_score(y_test, y_test_pred, average='weighted')}")
    print("=" * 40)

# Logistic Regression
evaluate_model(log_model, X_train, y_train, X_test, y_test, "Logistic Regression")

# Random Forest
evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")


# Gaussian Naive Bayes
evaluate_model(gnb_model, X_train, y_train, X_test, y_test, "Gaussian Naive Bayes")

# Decision Tree
evaluate_model(dt_model, X_train, y_train, X_test, y_test, "Decision Tree")


#plot for decision tree

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

# Initializing and training the decision tree classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# Plotting the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=data.columns[:-1], class_names=['0', '2'], rounded=True)
plt.title("Decision Tree for ramsomware Prediction")
plt.show()







from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Dictionary to store AUC scores
model_auc_scores = {}

# Prepare a figure for ROC curves
plt.figure(figsize=(14, 10))

# Loop through all classifiers
for name, model in models.items():
    try:
        # Some models like SVC need probability=True or use decision_function
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        # Compute AUC
        auc_score = roc_auc_score(y_test, y_proba)
        model_auc_scores[name] = auc_score

        # Compute ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

    except Exception as e:
        print(f"Skipping {name} due to error: {e}")

# Finalize ROC Curve Plot
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.title('ROC Curves of All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Plot AUC Score Distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=list(model_auc_scores.keys()), y=list(model_auc_scores.values()), palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.ylabel("AUC Score")
plt.title("AUC Score Distribution by Model")
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.show()

