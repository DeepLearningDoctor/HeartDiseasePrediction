from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Read the csv dataset file
df = pd.read_csv('heart.csv')
# Keep only relevant features
df = df[['age', 'trtbps', 'chol', 'output']]
# convert to numpy array
# nparr = np.array(df)
X = df.iloc[:, :-1].astype('int')
y = df.iloc[:, -1].astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Standard scaling the data
# sc_x = StandardScaler()
# sc_y = StandardScaler()

# Fit to data, then transform it.
# X_train = sc_x.fit_transform(np.asarray(X_train))
# y_train = sc_y.fit_transform(np.asarray(y_train).reshape(-1, 1))

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
test_prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, test_prediction)
# print("x:", X_train.head(2), "\ny:", y_train.head())
print("coeff:", model.coef_, "intercept:", model.intercept_,'test accuracy',accuracy)


@app.route('/')
def hello_world():
    return render_template('heartDiseaseForm.html')


@app.route('/predictpage', methods=['POST', 'GET'])
def predict():
    input = [int(x) for x in request.form.values()]
    input_arr = [np.array(input)]
    print('input', input)
    print('converted array:', input_arr)

    # For testing purpose create a list
    # prediction = [[1.2345, 2.3456]]
    prediction = model.predict(input_arr)
    prob = model.predict_proba(input_arr)
    print('type:', type(prediction), 'prediction:', prediction,'prob:',prob)

    if prob[0][0] >= prob[0][1]:
        return render_template('heartDiseaseForm.html', pred=f"You have high probability of having  heart disease")
        # output = f"You have high probability of having  heart disease, prob:{prob}"
    else:
        return render_template('heartDiseaseForm.html', pred=f"Relax.You probably don't have heart disease")
        # output = f"Relax.You probably don't have heart disease,prob{prob}"


if __name__ == '__main__':
    app.run(debug=True)
