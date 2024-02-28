# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/Sachin-vlr/basic-nn-model/assets/113497666/e89d2888-b469-47b3-8d48-5c98925cd0b9)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
DEVELOPED BY : SACHIN.C
REG NO : 212222230125
```
```PYTHON
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dep').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'INPUT':'float'})
dataset1 = dataset1.astype({'OUTPUT':'float'})

dataset1.head()

X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)
ai_brain=Sequential([Dense(7,activation="relu"),Dense(14,activation="relu"),Dense(1)])
ai_brain.compile(optimizer="rmsprop",loss="mse")
ai_brain.fit(X_train,y_train,epochs=3000)

loss=pd.DataFrame(ai_brain.history.history)
loss.plot()

x_test1=Scaler.transform(X_test)
ai_brain.evaluate(x_test1,y_test)

x_n1=[[11]]
x_n1_1=Scaler.transform(x_n1)

ai_brain.predict(x_n1_1)
```
## Dataset Information

![image](https://github.com/Sachin-vlr/basic-nn-model/assets/113497666/b1b00d6a-46be-41f5-b846-4e681fdb7682)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Sachin-vlr/basic-nn-model/assets/113497666/fd8ecc7a-41f0-4128-9af4-bedc2454243a)


### Test Data Root Mean Squared Error

![Screenshot 2024-02-28 110034](https://github.com/Sachin-vlr/basic-nn-model/assets/113497666/dd07f1dd-54e5-45ae-9052-fbc168731761)

### New Sample Data Prediction

![Screenshot 2024-02-28 105912](https://github.com/Sachin-vlr/basic-nn-model/assets/113497666/c1926452-53b9-41fe-a65f-21c0aecf5dac)

## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
