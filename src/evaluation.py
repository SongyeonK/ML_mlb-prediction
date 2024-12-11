from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

def evaluate_model(y_true, y_pred):
    """RMSE와 R-squared 계산."""
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R2": r2}

def plot_predictions(y_true, y_pred):
    """예측 결과 시각화."""
    plt.scatter(range(len(y_true)), y_true, alpha=0.5, label='Actual')
    plt.scatter(range(len(y_pred)), y_pred, alpha=0.5, label='Predicted', marker='x')
    plt.legend()
    plt.title("Prediction Results")
    plt.show()
