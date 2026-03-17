import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle #pkl
import os

def main():
    # 데이터 로드
    print("1. 데이터 로드")
    df = pd.read_csv("data/mission15_train.csv")

    # 데이터 전처리
    print("2. 데이터 전처리 및 X, y 분리")
    # 문자열(Yes/No) => 수치형(0, 1) 변환
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"No": 0,"Yes": 1})

    # X(특성), y(타겟) 분리
    X = df.drop('Performance Index', axis=1)
    y = df['Performance Index']

    # 선형 회귀 모델
    # 검증용 분할 없이 100% 데이터로 학습합니다.
    print("3. 모델 학습")
    model = LinearRegression()
    model.fit(X, y)

    # 예측
    y_pred = model.predict(X)

    # 성능 평가
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # 모델 저장
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"모델이 '{model_filename}'으로 저장되었습니다.")

if __name__ == "__main__":
    main()