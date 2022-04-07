def predict(x: float, w: float, b:float):
    return w * x + b

def update_w_and_b(x_data: list[float], y_data: list[float], w: float, b: float, alpha: float):
    N = len(x_data)
    dl_dw, dl_db = 0, 0
    for i in range(N):
        dl_dw += -2 * x_data[i] * (y_data[i] - (w * x_data[i] + b))
        dl_db += -2 * (y_data[i] - (w * x_data[i] + b))

    next_w = w - alpha * (1/N) * dl_dw
    next_b = b - alpha * (1/N) * dl_db
    return next_w, next_b

def train(x_data: list[float], y_data: list[float], w: float, b: float, alpha: float, epochs: int):
    for _ in range(epochs):
        w, b = update_w_and_b(x_data, y_data, w, b, alpha)
    return w, b

if __name__ == '__main__':
    x_data = [1 , 2, 3]
    y_data = [1, 2, 3]

    w_star, b_star = train(x_data=x_data, y_data=y_data, w=0.0, b=0.0, alpha=0.001, epochs=15000)
    new_x = 5
    print("prediction for x = 5: y =", predict(x=new_x, w=w_star, b=b_star))
