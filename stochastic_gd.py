import random

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

def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = list(zip(x, y))
    theta, alpha  = theta_0, alpha_0
    min_theta, min_value = None, float('inf')
    iterations_without_improvement = 0

    while iterations_without_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            min_theta, min_value = theta, value
            iterations_without_improvement = 0
            alpha = alpha_0
        else:
            iterations_without_improvement += 1
            alpha *= 0.9

        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta

if __name__ == '__main__':
    x_data = [1, 2, 3]
    y_data = [3, 5, 7]

    random.seed(0)
    theta = [random.random(), random.random()]

    w_star, b_star = minimize_stochastic(squared_error, squared_error_gradient, x_data, y_data, theta, 0.0001)
    x_new = 4
    y_new = predict(w_star, b_star, x_new)
    print("prediction for x =", x_new, ", y =", y_new)
