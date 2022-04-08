def update_w_and_b(x, y, w, b, alpha):
    N = len(x)
    dl_dw, dl_db = 0, 0
    for x_i, y_i in zip(x, y):
        dl_dw += -2 * x_i * (y_i - (w * x_i + b))
        dl_db += -2 * (y_i - (w * x_i + b))

    next_w = w - (alpha / N) * dl_dw
    next_b = b - (alpha / N) * dl_db
    return next_w, next_b

def train(x, y, w, b, alpha, epochs):
    for _ in range(epochs):
        w, b = update_w_and_b(x, y, w, b, alpha)
    return w, b

def predict(w, b, x_i):
    return w * x_i + b

if __name__ == '__main__':
    x_data = [1, 2, 3]
    y_data = [3, 5, 7]

    w_star, b_star = train(x_data, y_data, w=0.0, b=0.0, alpha=0.001, epochs=15000)
    x_new = 4
    y_new = predict(w_star, b_star, x_new)
    print("prediction for x =", x_new, ", y =", y_new)
