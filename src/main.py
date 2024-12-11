from src.data_preprocessing import load_data, check_missing_values, scale_features
from src.eda import plot_correlation_heatmap, plot_scatter
from src.modeling import train_model, predict
from src.evaluation import evaluate_model, plot_predictions

def main():
    # 데이터 로드
    df = load_data('data/mlb_ML.csv')
    
    # 데이터 전처리
    missing_ratio = check_missing_values(df)
    if missing_ratio.empty:
        print("No missing values found.")
    else:
        print("Missing values found:", missing_ratio)

    scale_columns = ['OPS', 'HR', 'BB', 'SO', 'ERA', 'HR9', 'SO9', 'SO/W']
    df = scale_features(df, scale_columns)

    # 데이터 분리
    X = df[scale_columns]
    y = df['W-L%']
    train_indices = df['Year'] < 2020
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[~train_indices], y[~train_indices]

    # 모델 학습
    model = train_model(X_train, y_train)

    # 예측 및 평가
    y_pred = predict(model, X_test)
    metrics = evaluate_model(y_test, y_pred)
    print("Model Performance:", metrics)

    # 결과 시각화
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()
