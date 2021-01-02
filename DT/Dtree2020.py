import numpy
import numpy as numpy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('brain_result_data.csv')
print(df.head())
X = df.iloc[:, 1:30].values
Y = df.iloc[:, 0].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = DecisionTreeClassifier(criterion="entropy",
                                  random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
import numpy as np
from sklearn import metrics
print("Decision Tree Model Accuracy(in %):", metrics.accuracy_score(Y_test, y_pred)*100)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
print()

ar=[1.4964519739151,-1.3190593719482422,-1.3917956352233887,-0.4932268261909485,0.04988037049770355,0.276285856962204,0.6570195555686951,0.7601263523101807,-0.7038455605506897,1.1973830461502075,-0.37654730677604675,-0.33329179883003235,0.19508500397205353,-0.8389440774917603,0.5898507237434387,-0.7905895113945007,-0.4106820821762085,0.12638969719409943,0.3677607774734497,-0.8598896861076355,-0.5588604807853699,0.10473707318305969,0.12262170761823654,0.3511357307434082,0.5570389628410339,-0.27849072217941284,-0.003169858129695058,-0.2968571186065674,-0.6430150866508484]
ar1=np.array(ar).reshape(1,-1)
print(classifier.predict(ar1))
print()