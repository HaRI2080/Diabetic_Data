# Diabetes Readmission Prediction

## 1. Project Overview

이 프로젝트는 당뇨 환자의 병원 입원 기록을 바탕으로 **30일 이내 재입원 여부**를 예측하는 머신러닝 프로젝트입니다.

환자의 인구학적 정보, 입원 정보, 검사 기록, 약물 사용 정보, 과거 의료 이용 이력 등을 활용하여 재입원 가능성을 예측합니다. Target variable은 `readmitted`이며, 다음과 같이 binary target으로 변환합니다.

- `readmitted == "<30"` -> `1`
- `readmitted == "NO"` 또는 `readmitted == ">30"` -> `0`

따라서 최종 문제 유형은 **binary classification**입니다.

## 2. Dataset

사용 데이터는 `diabetic_data.csv`입니다.

데이터 출처는 UCI Machine Learning Repository의 **Diabetes 130-US Hospitals for Years 1999-2008** 데이터셋입니다. 한 행은 한 번의 환자 입원 기록, 즉 patient encounter를 의미합니다.

주요 변수 예시는 다음과 같습니다.

| Variable | Description |
| --- | --- |
| `age` | 환자 연령대 |
| `gender` | 성별 |
| `race` | 인종 |
| `admission_type_id` | 입원 유형 |
| `discharge_disposition_id` | 퇴원 상태 |
| `admission_source_id` | 입원 경로 |
| `time_in_hospital` | 입원 기간 |
| `num_lab_procedures` | 검사 수 |
| `num_procedures` | 처치 수 |
| `num_medications` | 투약 수 |
| `number_outpatient` | 외래 방문 횟수 |
| `number_emergency` | 응급 방문 횟수 |
| `number_inpatient` | 과거 입원 횟수 |
| `number_diagnoses` | 진단 수 |
| `A1Cresult` | A1C 검사 결과 |
| `insulin` | 인슐린 처방 변화 |
| `diabetesMed` | 당뇨병 약물 사용 여부 |
| `readmitted` | 재입원 여부 |

## 3. Project Structure

```text
project/
├── diabetic_data.csv                         # 분석에 사용되는 원본 데이터
├── Logistic Regression.ipynb                 # Logistic Regression 및 K-Means cluster feature 비교
├── Random Forest.ipynb                       # Random Forest Classifier 모델
├── XGBoost.ipynb                             # XGBoost Classifier 모델
├── logistic_regression_results.csv           # 분석 실행 후 생성됨
├── random_forest_results.csv                 # 분석 실행 후 생성됨
├── xgboost_results.csv                       # 분석 실행 후 생성됨
└── README.md
```

각 노트북의 역할은 다음과 같습니다.

- `Logistic Regression.ipynb`: Logistic Regression 단독 모델과 K-Means cluster label을 추가한 Logistic Regression 모델을 비교합니다.
- `Random Forest.ipynb`: Random Forest Classifier를 이용해 30일 이내 재입원 여부를 예측합니다.
- `XGBoost.ipynb`: XGBoost Classifier를 이용해 30일 이내 재입원 여부를 예측합니다.

## 4. Preprocessing

전처리 과정은 모델 간 공정한 비교가 가능하도록 동일한 기준을 따릅니다.

- `readmitted` target 이진화
  - `"<30"` -> `1`
  - `"NO"`, `">30"` -> `0`
- `encounter_id`, `patient_nbr` 등 식별자 변수 제거
- `"?"` 값을 결측치로 처리
- `weight`, `payer_code`, `medical_specialty` 등 결측 비율이 높은 변수 제거
- Train/test split 적용
  - `test_size=0.2`
  - `random_state=42`
  - `stratify=y`
- 범주형 변수는 One-Hot Encoding 적용
- 수치형 변수는 median 등 적절한 값으로 결측치 처리
- Logistic Regression과 K-Means에는 `StandardScaler` 적용
- Random Forest와 XGBoost는 tree-based model이므로 scaling을 필수로 적용하지 않음
- 데이터 누수를 막기 위해 preprocessing은 train set에만 fit하고, test set에는 transform만 적용

## 5. Models

### Model 1: Logistic Regression

Logistic Regression은 기본 baseline 모델입니다. 선형 모델이므로 coefficient를 통해 각 feature가 예측에 미치는 방향성을 해석할 수 있습니다.

추가로 K-Means clustering으로 만든 cluster label을 feature로 넣은 Logistic Regression 모델과 비교합니다. 여기서 K-Means는 단독 예측 모델이 아니라 환자군 정보를 만드는 비지도학습 단계로 사용됩니다.

### Model 2: Random Forest

Random Forest는 여러 decision tree를 결합하는 tree-based ensemble model입니다. 비선형 관계와 변수 간 상호작용을 반영할 수 있으며, feature importance를 통해 예측에 중요한 변수를 확인할 수 있습니다.

### Model 3: XGBoost

XGBoost는 boosting 기반 tree model입니다. 이전 tree가 잘 예측하지 못한 부분을 다음 tree가 보완하면서 성능을 개선하는 방식입니다.

클래스 불균형을 고려하기 위해 `scale_pos_weight`를 사용할 수 있으며, feature importance를 통해 예측에 중요한 변수를 확인할 수 있습니다.

## 6. Evaluation Metrics

모델 평가는 같은 test set에서 아래 지표를 사용해 수행합니다.

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

의료 재입원 예측 문제에서는 Accuracy만으로 모델을 판단하기 어렵습니다. 실제 30일 이내 재입원 환자를 놓치는 것이 중요할 수 있으므로 **Recall**, **F1-score**, **ROC-AUC**도 함께 비교합니다.

## 7. How to Run

1. 필요한 라이브러리를 설치합니다.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

2. `diabetic_data.csv` 파일을 프로젝트 폴더에 둡니다.

3. 아래 notebook을 순서대로 실행합니다.

- `Logistic Regression.ipynb`
- `Random Forest.ipynb`
- `XGBoost.ipynb`

4. 각 notebook에서 생성된 결과 CSV 파일을 바탕으로 모델 성능을 비교합니다.

## 8. Results

최종 성능은 각 notebook을 실행한 뒤 생성되는 결과 파일을 기준으로 정리합니다.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| K-Means + Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |

## 9. Interpretation

모델 해석은 coefficient 또는 feature importance를 통해 수행할 수 있습니다.

- Logistic Regression coefficient는 양수/음수 방향성을 해석할 수 있습니다.
- Random Forest와 XGBoost의 feature importance는 방향성이 아니라 예측에 사용된 중요도를 의미합니다.
- 예를 들어 `number_inpatient`, `number_emergency`, `time_in_hospital`, `num_medications`, `number_diagnoses` 등이 중요하게 나타난다면 과거 의료 이용 이력, 입원 기간, 투약 수, 진단 복잡도 등이 30일 이내 재입원 예측에 관련성이 있는 변수로 해석할 수 있습니다.
- 의료적 해석에서는 특정 변수가 재입원을 직접 유발한다고 단정하지 않고, 예측에 관련성이 있는 변수로 표현해야 합니다.

## 10. Notes

- 이 프로젝트는 교육 목적의 데이터 분석 프로젝트입니다.
- 모델 결과는 실제 임상 의사결정을 직접 대체할 수 없습니다.
- 데이터는 1999-2008년 미국 병원 데이터를 기반으로 하므로, 현재 한국 임상 환경에 그대로 일반화하기에는 한계가 있습니다.
- 재입원 예측 결과는 인과관계가 아니라 통계적/기계학습적 예측 결과로 해석해야 합니다.
