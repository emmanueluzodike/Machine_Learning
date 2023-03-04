import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


from LinearRegression import LinearRegression


def main():
    # DATASET hosted on utd web server
    url = 'http://www.utdallas.edu/~emu200000/Folds5x2_pp.xlsx'
    # Read the XLSX file into a pandas dataframe
    df = pd.read_excel(url)
    # Display the first 5 rows of the dataframe
    # print(df.head())

    # Remove null or NA values
    df = df.dropna()

    # Remove redundant rows
    df = df.drop_duplicates()

    # print(df)

    # make a copy of the dataframe and drop the PE column(our target)
    X = df.drop(['PE'], axis=1)

    # print(X)

    # Stardardize or normalize the data
    col = X.columns
    s = StandardScaler()  # makes every attribute a normal dist with mean = 0 and varience 1
    X = pd.DataFrame(s.fit(X).fit_transform(X), columns=col)

    # print(X)

    # set the target variable
    Y = df['PE']

    # print(Y)

    # split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    # create model
    model = LinearRegression(0.001, 1000000, 0.000001)

    # train model
    model.gradient_descent(X_train, Y_train)

    # model evaluation for training set
    test_model(X_train, Y_train, model, "Training")

    # model evaluation for testing set
    test_model(X_test, Y_test, model, "Testing")

    # create plots
    plot_graph_of_test_data(X_test, Y_test, model)


def test_model(X, Y, model, set_type):
    y_test_predict = model.predict(X)
    rmse = (np.sqrt(mean_squared_error(Y, y_test_predict)))
    r2 = r2_score(Y, y_test_predict)

    print("\nThe model performance for " + set_type + " set")
    print("--------------------------------------")
    print('RMSE is {}'.format(rmse))
    print('R2 score is {}'.format(r2))
    print('Mean Squared Error :', mean_squared_error(Y, model.predict(X)))
    print('Mean Absolute Error :', mean_absolute_error(Y, model.predict(X)))
    print("\n")


def plot_graph_of_test_data(X_test, Y_test, model):
    # TARGET VS PREDICTED
    # print("TARGET VS PREDICTED")
    # model evaluation for testing set
    plt.scatter(Y_test, model.predict(X_test))
    plt.grid()
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.title('scatter plot between actual y and predicted y')
    plt.show()

    # print("\n Target vs AT")
    y_pred = model.predict(X_test)
    # Plot the predictions against the feature
    # This graph shows the predicted values against the actual values for the feature AT
    plt.scatter(X_test['AT'], Y_test, color='blue')  # expected value
    plt.scatter(X_test['AT'], y_pred, color='red')  # predicted value
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.show()

    # MSE VS ITERATION
    # Keep track of the MSE at each iteration
    # print("\n MSE VS ITERATION")
    mse = []
    for i in range(X_test.shape[0]):
        y_pred = model.predict(X_test)
        mse.append(mean_squared_error(Y_test, y_pred))

    # Plot the MSE versus the number of iterations
    plt.plot(range(X_test.shape[0]), mse)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
