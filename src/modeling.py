from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    """선형 회귀 모델을 학습합니다."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """테스트 데이터를 예측합니다."""
    return model.predict(X_test)
