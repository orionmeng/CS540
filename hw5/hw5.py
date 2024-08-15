import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def plot_data(data):
    years = []
    days = []
    for row in data:
        y = int(row['year'].split('-')[0].strip('"'))
        d = row['days'].strip('"')
        if d:
            years.append(y)
            days.append(int(d))
    plt.plot(years, days, linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.title('Year vs. Number of Frozen Days')
    plt.savefig("plot.jpg")

def augment_features(data):
    X = []
    Y = []
    for row in data:
        year = int(row['year'].split('-')[0].strip('"'))
        days = row['days'].strip('"')
        if days:
            X.append([1, year])
            Y.append(int(days))
    return np.array(X), np.array(Y)

def compute_beta(X, Y):
    X_transpose = np.transpose(X)
    Z = np.dot(X_transpose, X)
    I = np.linalg.inv(Z)
    P_I = np.dot(I, X_transpose)
    beta_hat = np.dot(P_I, Y)
    return beta_hat

def predict_y_test(beta_hat, x_test):
    return beta_hat[0] + (beta_hat[1] * x_test)

def predict_x_star(beta_hat):
    return -beta_hat[0] / beta_hat[1]

def sign(beta_hat):
    symbol = "="
    if beta_hat[1] > 0:
        symbol = ">"
    elif beta_hat[1] < 0:
        symbol = "<"
    return symbol

def discuss_symbol():
    statement = "The sign of β1 indicates the relationship between the year and the days of ice cover."
    statement += " If positive (>), then as the year increases, the days of ice cover tends to increase."
    statement += " If negative (<), then as the year increases, the days of ice cover tends to decrease."
    statement += " If zero (=), no clear linear relationship between the year and the days of ice cover."
    return statement

def discuss_x_star(x_star, data):
    last_year = int(data[-1]['year'].split('-')[0].strip('"'))
    statement = ""
    if x_star > last_year:
        statement += "The prediction is that Lake Mendota will stop freezing beyond the scope of the dataset. "
        statement += "This could be a compelling prediction given that the linear trend persists."
    else:
        statement += "The prediction is that Lake Mendota will stop freezing within the scope of the dataset. "
        statement += "This is not a compelling prediction since it contradicts the provided data."
    return statement

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 hw5.py filename.csv")
    else:
        filename = sys.argv[1]
        data = read_csv(filename)
        X, Y = augment_features(data)
        Z = np.dot(X.T, X)
        I = np.linalg.inv(Z)
        PI = np.dot(I, X.T)
        beta_h = compute_beta(X, Y)
        x_test = 2022
        y_test = predict_y_test(beta_h, x_test)
        symbol = sign(beta_h)
        x_star = predict_x_star(beta_h)
        stmt_1 = discuss_symbol()
        stmt_2 = discuss_x_star(x_star, data)

        plot_data(data)

        print("Q3a:")
        print(X.astype(np.int64))

        print("Q3b:")
        print(Y.astype(np.int64))

        print("Q3c:")
        print(Z.astype(np.int64))

        print("Q3d:")
        print(I)

        print("Q3e:")
        print(PI)

        print("Q3f:")
        print(beta_h)

        print("Q4: " + str(y_test))

        print("Q5a: " + symbol)
        print("Q5b: " + stmt_1)

        print("Q6a: " + str(x_star))
        print("Q6b: " + stmt_2)
