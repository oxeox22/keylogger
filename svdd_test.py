import numpy as np
import sqlite3
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
import plotly.graph_objects as go

# 데이터베이스에서 데이터를 로드하는 함수
def load_data_from_db(database_path, table_name, feature_columns):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    query = f"SELECT {', '.join(feature_columns)} FROM {table_name}"
    cursor.execute(query)
    data = cursor.fetchall()
    X = np.array(data)
    conn.close()
    return X

class SVDD:
    def __init__(self, C=0.1, s=10):
        self.C = C ## regularization parameter
        self.s = s ## parameter for rbf kernel
        self.alpha = None ## alpha
        self.R2 = None ## radius
        self.Q = None ## kernel matrix
        self.X = None
        self.support_vectors_ = None
    
    def rbf_kernel(self, x, y, s):
        return np.exp(-np.sum(np.square(x-y))/(s**2))
    
    def make_kernel_matrix(self, X, s):
        n = X.shape[0]
        Q = np.zeros((n,n))
        q_list = []
        ##커널 값 계산
        for i in range(n):
            for j in range(i, n):
                q_list.append(self.rbf_kernel(X[i, ], X[j, ], s))
        Q_idx = np.triu_indices(len(Q))
        Q[Q_idx] = q_list
        Q = Q.T + Q - np.diag(np.diag(Q))
        return Q
    
    def fit(self, X): ##svdd fit시키기
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
        alphas = alphas.flatten() ##서포트벡터 확인
        S = ((alphas>1e-3)&(alphas<C))
        S_idx = np.where(S)[0]
        #서포트벡터 도메인 반지름 계산
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
    
    def predict(self, X): #데이터 포인트 label predict시킴
        return np.array([np.sign(self._predict(x)) for x in X])
    
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
train_database_path = r"C:\Users\user\Desktop\yena_fast_typing_data.db"
train_table_name = 'typing_data'
train_feature_columns = ['duration', 'duration2', 'duration3']
X_train = load_data_from_db(train_database_path, train_table_name, train_feature_columns)

# 테스트 데이터 로드
test_database_path = r"C:\Users\user\Desktop\yena_slow_typing_data.db"
test_table_name = 'typing_data'
test_feature_columns = ['duration', 'duration2', 'duration3']
X_test = load_data_from_db(test_database_path, test_table_name, test_feature_columns)

# 데이터 전처리: 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

print("중심:", center)
print("반지름:", radius)
print("평균 거리 (트레인 데이터셋):", train_mean_distance)
print("평균 거리 (테스트 데이터셋):", test_mean_distance)

# 그래프 생성
fig = go.Figure()

# 학습 데이터 플로팅
fig.add_trace(go.Scatter3d(
    x=X_train_scaled[:, 0],
    y=X_train_scaled[:, 1],
    z=X_train_scaled[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
    ),
    name='Training Data'
))

# 테스트 데이터 플로팅
fig.add_trace(go.Scatter3d(
    x=X_test_scaled[:, 0],
    y=X_test_scaled[:, 1],
    z=X_test_scaled[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='red',
    ),
    name='Test Data'
))

# 결정 경계를 나타내는 3차원 원 추가
#center = np.zeros(3) # 중심은 임의로 설정
radius = np.sqrt(svdd_model.R2) # 반지름은 계산된 반지름 사용
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Reds', opacity=0.3, name='Decision Boundary'))

# 그래프 레이아웃 설정 (너비: 1000px, 높이: 1000px)
fig.update_layout(
    scene=dict(
        xaxis=dict(title='Duration'),
        yaxis=dict(title='Duration2'),
        zaxis=dict(title='Duration3'),
    ),
    width=1000,
    height=1000
)

# 그래프 출력
fig.show()
