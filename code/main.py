import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="반도체 클러스터 청년 인구와 집값 분석",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ 반도체 클러스터의 청년 인구 유입과 아파트 매매가 회귀 분석")
st.markdown("""
**핵심 질문:** *반도체 클러스터(평택·용인·화성·이천)의 청년 인구 유입 증가는 정말 해당 지역의 아파트 매매가 상승을 이끄는 핵심 요인인가?*
- **출처:** 통계청 KOSIS 국가통계포털 (`2015.01 ~ 2024.12`, 4개 시 × 120개월 = 480행)
""")
st.divider()

TARGET_CITIES = ["평택시", "용인시", "화성시", "이천시"]


# 2. KOSIS 원본 CSV 자동 읽기 및 전처리 함수
def read_uploaded_csv(uploaded_file):
    for enc in ["cp949", "utf-8-sig", "utf-8", "euc-kr"]:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"파일 인코딩을 읽을 수 없습니다: {uploaded_file.name}")


def find_city_col(df):
    """데이터프레임의 모든 앞쪽 열 중에서 '평택시/용인시/화성시/이천시'가 들어있는 열을 자동으로 찾습니다."""
    for col in df.columns:
        if re.match(r"^\d{4}\.\d{2}", str(col).strip()):
            continue
        series_clean = df[col].astype(str).str.strip()
        if series_clean.isin(TARGET_CITIES).any():
            return col
    return None


def melt_dates(df, id_cols, value_name):
    date_cols = [c for c in df.columns if re.match(r"^\d{4}\.\d{2}", str(c).strip())]
    melted = df.melt(id_vars=id_cols, value_vars=date_cols, var_name="기준연월", value_name=value_name)
    melted["기준연월"] = melted["기준연월"].str.extract(r"^(\d{4}\.\d{2})")[0]
    melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
    return melted


def merge_kosis_files(files):
    df_pop_final, df_mig_final, df_apt_final = None, None, None

    for f in files:
        raw = read_uploaded_csv(f)
        fname = f.name
        cols_str = " ".join(map(str, raw.columns[:6]))
        sample_str = " ".join(raw.iloc[:20, :5].astype(str).values.flatten())

        city_col = find_city_col(raw)
        if city_col is None:
            raise ValueError(f"[{fname}] 파일에서 '평택시, 용인시, 화성시, 이천시' 열을 찾지 못했습니다.")

        raw[city_col] = raw[city_col].astype(str).str.strip()
        raw = raw[raw[city_col].isin(TARGET_CITIES)].copy()

        # (1) 주민등록인구 파일 판별
        if "주민등록" in fname or "연령" in cols_str or "20세" in sample_str or "20~24세" in sample_str or "20 - 24세" in sample_str:
            # 연령이 들어있는 열 탐색
            age_col = None
            for col in raw.columns:
                if col == city_col or re.match(r"^\d{4}\.\d{2}", str(col).strip()):
                    continue
                if raw[col].astype(str).str.strip().isin(["계", "20세", "20~24세", "20 - 24세"]).any():
                    age_col = col
                    break
            if age_col is None:
                raise ValueError(f"[{fname}] 파일에서 '연령별' 열을 찾지 못했습니다.")

            melted = melt_dates(raw, [city_col, age_col], "인구수").rename(
                columns={city_col: "지역명", age_col: "연령"}
            )
            melted["연령"] = melted["연령"].astype(str).str.strip()

            total_pop = melted[melted["연령"] == "계"][["지역명", "기준연월", "인구수"]].rename(
                columns={"인구수": "총인구수"}
            )
            youth_ages = (
                [f"{a}세" for a in range(20, 40)]
                + ["20~24세", "25~29세", "30~34세", "35~39세", "20 - 24세", "25 - 29세", "30 - 34세", "35 - 39세"]
            )
            youth_pop = (
                melted[melted["연령"].isin(youth_ages)]
                .groupby(["지역명", "기준연월"], as_index=False)["인구수"]
