def update_w_and_b(x_data: list[float], y_data: list[float], w: float, b: float, alpha: float):
    N = len(x_data)
    dl_dw, dl_db = 0, 0
    for x_i, y_i in zip(x_data, y_data):
        dl_dw += -2 * x_i * (y_i - (w * x_i + b))
        dl_db += -2 * (y_i - (w * x_i + b))

    next_w = w - (alpha / N) * dl_dw
    next_b = b - (alpha / N) * dl_db
    return next_w, next_b

def train(x_data: list[float], y_data: list[float], w: float, b: float, alpha: float, epochs: int):
    for _ in range(epochs):
        w, b = update_w_and_b(x_data=x_data, y_data=y_data, w=w, b=b, alpha=alpha)
    return w, b

def predict(x: float, w: float, b:float) -> float:
    return w * x + b

if __name__ == '__main__':
    x_data = [1, 2, 3]
    y_data = [3, 5, 7]

    w_star, b_star = train(x_data=x_data, y_data=y_data, w=0.0, b=0.0, alpha=0.001, epochs=15000)
    x_new = 4
    y_new = predict(x=x_new, w=w_star, b=b_star)
    print("prediction for x =", x_new, ", y =", y_new)
