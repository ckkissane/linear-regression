import random

from stochastic_gradient_descent import minimize_stochastic


def predict(w, b, x_i):
    return w * x_i + b


def error(w, b, x_i, y_i):
    return y_i - predict(w, b, x_i)


def squared_error(x_i, y_i, theta):
    w, b = theta
    return error(w, b, x_i, y_i) ** 2


def squared_error_gradient(x_i, y_i, theta):
    w, b = theta
    return [-2 * x_i * error(w, b, x_i, y_i), -2 * error(w, b, x_i, y_i)]


if __name__ == "__main__":
    x_data = [1, 2, 3]
    y_data = [3, 5, 7]

    random.seed(0)
    theta = [random.random(), random.random()]

    w_star, b_star = minimize_stochastic(
        squared_error, squared_error_gradient, x_data, y_data, theta, 0.0001
    )
    x_new = 4
    y_new = predict(w_star, b_star, x_new)
    print("prediction for x =", x_new, ", y =", y_new)
