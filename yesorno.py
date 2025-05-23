import numpy as np
import sqlite3
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox
from cvxopt import matrix, solvers

#데이터베이스에서 데이터를 불러오는 함수 정의
def load_data_from_db(database_path, table_name, feature_columns):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    query = f"SELECT {', '.join(feature_columns)} FROM {table_name}"
    cursor.execute(query)
    data = cursor.fetchall()
    X = np.array(data)
    conn.close()
    return X

#SVDD 모델 정의
class SVDD:
    def __init__(self, C=0.1, s=10):
        self.C = C #정규화 파라미터
        self.s = s #rbf 커널 파라미터
        self.alpha = None #알파 초기화
        self.R2 = None #반지름 초기화
        self.Q = None #커널 행렬 초기화
        self.X = None #훈련 데이터 초기화
        self.support_vectors_ = None #지원 벡터 초기화

#rbf 커널 함수 정의(x와 y 사이의 거리를 이용해 계산)
    def rbf_kernel(self, x, y, s):
        return np.exp(-np.sum(np.square(x-y))/(s**2))
    
#커널 행렬 생성
    def make_kernel_matrix(self, X, s):
        n = X.shape[0]
        Q = np.zeros((n,n))
        q_list = []
        for i in range(n):
            for j in range(i, n):
                q_list.append(self.rbf_kernel(X[i, ], X[j, ], s))
        Q_idx = np.triu_indices(len(Q))
        Q[Q_idx] = q_list
        Q = Q.T + Q - np.diag(np.diag(Q))
        return Q

#모델 학습 함수 정의
    def fit(self, X):
        C = self.C
        s = self.s
        Q = self.make_kernel_matrix(X, s)
        n = X.shape[0]
        P = matrix(2*Q)
        q = np.array([self.rbf_kernel(x, x, s) for x in X])
        q = matrix(q.reshape(-1, 1))
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
        h = matrix(np.hstack([np.zeros(n), np.ones(n)*C]))
        A = matrix(np.ones((1, n)))
        b = matrix(np.ones(1))
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])
        
        rho = 0
        alphas = alphas.flatten()
        S = ((alphas>1e-3)&(alphas<C))
        S_idx = np.where(S)[0]
        
        R2 = alphas.dot(Q.dot(alphas))
        
        for si in S_idx:
            temp_vector = np.array([-2*alphas[i]*self.rbf_kernel(X[i, ], X[si, ], s) for i in range(n)])
            R2 += (self.rbf_kernel(X[si, ], X[si, ], s) + np.sum(temp_vector))/len(S_idx)
        
        self.R2 = R2
        self.alphas = alphas
        self.X = X
        self.Q = Q
        self.support_vectors_ = X[S_idx]
        return self

#이상 탐지 함수 정의
    def predict(self, X):
        return np.array([np.sign(self._predict(x)) for x in X])

#예측 내부 함수 정의
    def _predict(self, x):
        X = self.X
        n = X.shape[0]
        alphas = self.alphas
        R2 = self.R2
        s = self.s
        Q = self.Q
        
        first_term = self.rbf_kernel(x, x, s)
        second_term = np.sum([2 * alphas[i] * self.rbf_kernel(x, X[i, ], s) for i in range(n)])
        thrid_term = alphas.dot(Q.dot(alphas))
        
        return R2-first_term+second_term-thrid_term


# 학습 데이터 로드
train_database_path = r"C:\Users\jh052\Desktop\SWUFORCE\제4회 교내연합세미나\keylogger\seol_slow_typing_data.db"
train_table_name = 'typing_data'
train_feature_columns = ['duration', 'duration2', 'duration3']
X_train = load_data_from_db(train_database_path, train_table_name, train_feature_columns)

# 테스트 데이터 로드
test_database_path = r"C:\Users\jh052\Desktop\SWUFORCE\제4회 교내연합세미나\keylogger\juhyeon_fast_typing_data.db"
test_table_name = 'typing_data'
test_feature_columns = ['duration', 'duration2', 'duration3']
X_test = load_data_from_db(test_database_path, test_table_name, test_feature_columns)

# 데이터 전처리: 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 데이터 스케일링
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)

scaler_test = StandardScaler()
X_test_scaled = scaler_test.fit_transform(X_test)

# SVDD 모델 훈련
svdd_model = SVDD(C=0.1, s=10)
svdd_model.fit(X_train_scaled)

# 원의 중심과 반지름 계산
center = svdd_model.support_vectors_.mean(axis=0)
radius = np.mean([np.linalg.norm(svdd_model.support_vectors_[i] - center) for i in range(len(svdd_model.support_vectors_))])

# 각 점들과 중심 사이의 거리 계산
def distance(point, center):
    return np.linalg.norm(point - center)

# 트레인 데이터셋의 각 점들과 중심 사이의 거리 계산
train_distances = [distance(point, center) for point in X_train_scaled]

# 테스트 데이터셋의 각 점들과 중심 사이의 거리 계산
test_distances = [distance(point, center) for point in X_test_scaled]

# 각 데이터셋의 평균 거리 계산
train_mean_distance = np.mean(train_distances)
test_mean_distance = np.mean(test_distances)


# 팝업 함수
def show_popup():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("알림", "접근이 차단되었습니다")

    # 일정 시간이 지난 후에 재귀적으로 팝업을 다시 띄움
    root.after(100, show_popup)  # 100밀리초 = 0.1초 후에 팝업을 다시 띄움

# 거리 차이가 0.02이상인 경우 팝업 표시
if abs(train_mean_distance - test_mean_distance) >= 0.02:
    root = tk.Tk()
    root.withdraw()
    root.after(500, show_popup)
    root.mainloop()
