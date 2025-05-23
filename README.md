# Keylogger
> Python 기반 키스트로크 다이나믹스 수집·분석 및 SVDD 이상 탐지 도구
> 
> 사용자 입력 패턴 분석을 통해 동일 인물이 아닌 사람이 사용할 경우 접근 차단

## 🚀 주요 기능
- **키스트로크 타이밍 수집**  
  - 키의 눌림 시간(hold time), 전후 키 릴리즈 간격(flight time) 등을 실시간 기록  
  - `keylogger+svdd+esc+yesorno.py` 에서 `pynput` 사용  
- **데이터베이스 저장**  
  - SQLite(`typing_data.db`)에 타이핑 데이터 및 백스페이스 통계 자동 삽입  
- **특징 벡터 생성**  
  - `NofingerFeature_d2.py` 모듈로 주요 타이핑 특성(hold, flight, error rate) 벡터화  
- **SVDD 기반 이상 탐지**  
  - `svdd_test.py` 로 SVDD(One-Class SVM) 모델 학습·테스트  
- **사용자 인터랙션**  
  - Enter 키 누르면 `yesorno.py` 호출, 간단한 Yes/No 프롬프트 제공  

## 📦 기술 스택 & 의존성
- Python 3.8+  
- SVDD
