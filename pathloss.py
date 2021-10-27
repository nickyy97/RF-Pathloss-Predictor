import pandas
from sklearn import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from flask import Flask , render_template, request
from flask import jsonify

app = Flask(__name__)

df = pandas.read_csv('dataset.csv')

X = df[['Distance', 'Frequency']]
y = df['Pathloss']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred_lr = regr.predict(X_test)

MAE_lr = metrics.mean_absolute_error(y_test,y_pred_lr)
MSE_lr = metrics.mean_squared_error(y_test, y_pred_lr)

knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

MAE_knn = metrics.mean_absolute_error(y_test,y_pred_knn)
MSE_knn = metrics.mean_squared_error(y_test, y_pred_knn)

def prediction_lr(Distance,Frequency):
    input_data = [[Distance,Frequency]]
    predict = regr.predict(input_data)
    # print(predict)
    return predict

def prediction_knn(Distance,volume):
    input_data = [[Distance,Frequency]]
    predict = knn_model.predict(input_data)
    # print(predict)
    return predict

@app.route('/')
def home():
    return render_template('index.html', prediction=0)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        distance = request.form['distance']
        frequency = request.form['frequency']
        pred_lr = prediction_lr(distance, frequency)
        pred_knn = prediction_knn(distance, frequency)
        predint_lr = pred_lr[0]
        predint_knn = pred_knn[0]
        print(predint_lr, predint_knn)
        return render_template('index.html', prediction_lr=predint_lr,  prediction_knn=predint_knn, mae_lr=MAE_lr,  mse_lr=MSE_lr, mae_knn=MAE_knn,  mse_knn=MSE_knn)
    else:
        return render_template('index.html', prediction=1, prediction_lr="Something went wrong", prediction_knn="Something went wrong", mae_lr=0,  mse_knn=0)

if __name__ == '__main__':
    app.run(debug=True)

