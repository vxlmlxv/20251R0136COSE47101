import pandas as pd
import glob
import logging
from tqdm import tqdm
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ⏱ 시작 시각 표시
print("⏱ 시작 시각:", datetime.now().strftime("%H:%M:%S"))

# 📝 로그 파일 설정
logging.basicConfig(filename='log_pattern_mining.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("패턴 마이닝 시작")

# 📁 민원 발생 데이터 불러오기
positive_files = glob.glob("result_chunks/results_part_*.csv")
dfs = []

print(f"📂 민원 발생 파일 {len(positive_files)}개 불러오는 중...")
for file in tqdm(positive_files, desc="Loading"):
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        logging.error(f"파일 {file} 읽기 실패: {e}")

# 🔗 전체 병합
df = pd.concat(dfs, ignore_index=True)
logging.info(f"총 데이터 개수: {len(df)}")

# ✅ 필요한 변수 선택 및 결측 제거
cols = ['총주차면수', '평균요금', '평균운영시간', 'CCTV개수']
df = df[cols].dropna()
logging.info("결측 제거 완료")

# ▶️ 변수별 사분위수 기반 범주화
df['주차면_구간'] = pd.qcut(df['총주차면수'], q=4, labels=['매우적음', '적음', '보통', '많음'])
df['요금_구간'] = pd.qcut(df['평균요금'], q=4, labels=['매우저렴', '저렴', '보통', '비쌈'])
df['운영시간_구간'] = pd.qcut(df['평균운영시간'], q=4, labels=['짧음', '보통', '김', '매우김'])
df['CCTV_구간'] = pd.qcut(df['CCTV개수'], q=4, labels=['적음', '보통', '많음', '매우많음'])
logging.info("사분위수 기반 범주화 완료")

# ▶️ 트랜잭션 리스트 생성
transactions = df[['주차면_구간', '요금_구간', '운영시간_구간', 'CCTV_구간']].astype(str).values.tolist()
logging.info("트랜잭션 데이터 생성 완료")

# ▶️ One-hot encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_tf = pd.DataFrame(te_ary, columns=te.columns_)
logging.info("One-hot encoding 완료")

# ▶️ Apriori로 빈발 항목집합 추출
frequent_itemsets = apriori(df_tf, min_support=0.05, use_colnames=True)
logging.info(f"빈발 항목집합 개수: {len(frequent_itemsets)}")

# ▶️ 연관 규칙 추출
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
logging.info(f"추출된 연관 규칙 수: {len(rules)}")

# ▶️ 결과 저장
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_csv("pattern_mining_results.csv", index=False)
print("✅ 연관 규칙 저장 완료: pattern_mining_results.csv")
logging.info("연관 규칙 저장 완료")

print("⏱ 종료 시각:", datetime.now().strftime("%H:%M:%S"))
