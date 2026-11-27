import os
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
st.set_page_config(page_title="반도체 클러스터 청년 인구와 집값 분석", page_icon="🏙️", layout="wide")

st.title("🏙️ 반도체 클러스터의 청년 인구 유입과 아파트 매매가 회귀 분석")
st.markdown("""
**핵심 질문:** *반도체 클러스터(평택·용인·화성·이천)의 청년 인구 유입 증가는 정말 해당 지역의 아파트 매매가 상승을 이끄는 핵심 요인인가?*
- **출처:** 통계청 KOSIS 국가통계포털 (`2015.01 ~ 2024.12`, 4개 시 × 120개월 = 480행)
""")
st.divider()

TARGET_CITIES = ["평택시", "용인시", "화성시", "이천시"]
CITY_NORMALIZE_MAP = {
    "평택": "평택시", "평택시": "평택시",
    "용인": "용인시", "용인시": "용인시",
    "화성": "화성시", "화성시": "화성시",
    "이천": "이천시", "이천시": "이천시",
}


def read_uploaded_csv(uploaded_file):
    for enc in ["cp949", "utf-8-sig", "utf-8", "euc-kr"]:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"파일 인코딩을 읽을 수 없습니다: {uploaded_file.name}")


def find_city_col(df):
    for col in df.columns:
        if re.match(r"^\d{4}\.\d{2}", str(col).strip()):
            continue
        if df[col].astype(str).str.strip().map(CITY_NORMALIZE_MAP).notna().any():
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
            raise ValueError(f"[{fname}] 파일에서 대상 도시 열을 찾지 못했습니다.")

        raw[city_col] = raw[city_col].astype(str).str.strip().map(CITY_NORMALIZE_MAP)
        raw = raw[raw[city_col].isin(TARGET_CITIES)].copy()

        # (1) 주민등록인구
        if "주민등록" in fname or "연령" in cols_str or "20세" in sample_str or "20~24세" in sample_str or "20 - 24세" in sample_str:
            age_col = None
            for col in raw.columns:
                if col == city_col or re.match(r"^\d{4}\.\d{2}", str(col).strip()):
                    continue
                if raw[col].astype(str).str.strip().isin(["계", "20세", "20~24세", "20 - 24세"]).any():
                    age_col = col
                    break
            melted = melt_dates(raw, [city_col, age_col], "인구수").rename(columns={city_col: "지역명", age_col: "연령"})
            melted["연령"] = melted["연령"].astype(str).str.strip()

            total_pop = melted[melted["연령"] == "계"][["지역명", "기준연월", "인구수"]].rename(columns={"인구수": "총인구수"})
            youth_ages = [f"{a}세" for a in range(20, 40)] + ["20~24세", "25~29세", "30~34세", "35~39세", "20 - 24세", "25 - 29세", "30 - 34세", "35 - 39세"]
            youth_pop = melted[melted["연령"].isin(youth_ages)].groupby(["지역명", "기준연월"], as_index=False)["인구수"].sum().rename(columns={"인구수": "청년인구수"})

            df_pop_final = pd.merge(total_pop, youth_pop, on=["지역명", "기준연월"])
            df_pop_final["청년인구비율"] = (df_pop_final["청년인구수"] / df_pop_final["총인구수"] * 100).round(2)
            df_pop_final = df_pop_final[["지역명", "기준연월", "총인구수", "청년인구비율"]]

        # (2) 아파트 매매가격지수
        elif "매매가격" in fname or "아파트" in sample_str or "주택유형" in cols_str:
            for col in raw.columns:
                if col == city_col or re.match(r"^\d{4}\.\d{2}", str(col).strip()):
                    continue
                if raw[col].astype(str).str.strip().eq("아파트").any():
                    raw = raw[raw[col].astype(str).str.strip() == "아파트"]
                    break
            df_apt_final = melt_dates(raw, [city_col], "아파트매매가격지수").rename(columns={city_col: "지역명"})
            df_apt_final = df_apt_final.groupby(["지역명", "기준연월"], as_index=False)["아파트매매가격지수"].first()

        # (3) 순이동인구수
        else:
            for col in raw.columns:
                if col == city_col or re.match(r"^\d{4}\.\d{2}", str(col).strip()):
                    continue
                if raw[col].astype(str).str.contains("순이동").any():
                    raw = raw[raw[col].astype(str).str.contains("순이동")]
                    break
            df_mig_final = melt_dates(raw, [city_col], "순이동인구수").rename(columns={city_col: "지역명"})
            df_mig_final = df_mig_final.groupby(["지역명", "기준연월"], as_index=False)["순이동인구수"].first()

    if df_pop_final is None or df_mig_final is None or df_apt_final is None:
        raise ValueError("인구, 이동자수, 매매가격지수 3개 파일을 모두 업로드해 주세요.")

    merged = pd.merge(df_pop_final, df_mig_final, on=["지역명", "기준연월"], how="inner")
    merged = pd.merge(merged, df_apt_final, on=["지역명", "기준연월"], how="inner")
    merged = merged[(merged["기준연월"] >= "2015.01") & (merged["기준연월"] <= "2024.12")]
    before_rows = len(merged)
    merged = merged.sort_values(["지역명", "기준연월"]).dropna().reset_index(drop=True)
    return merged, before_rows


@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2024-12-01", freq="MS").strftime("%Y.%m")
    rows = []
    for city in TARGET_CITIES:
        base_pop = {"평택시": 450000, "용인시": 960000, "화성시": 540000, "이천시": 205000}[city]
        base_youth = {"평택시": 28.5, "용인시": 27.0, "화성시": 31.0, "이천시": 27.5}[city]
        for i, d in enumerate(dates):
            total_pop = int(base_pop + i * np.random.uniform(800, 1800))
            youth_ratio = round(base_youth + (i * 0.015) + np.random.normal(0, 0.3), 2)
            net_migration = int(np.random.normal(800, 450))
            apt_index = round(65.0 + (youth_ratio - 26.0) * 4.2 + (net_migration * 0.004) + (i * 0.18) + np.random.normal(0, 2.5), 2)
            rows.append([city, d, total_pop, youth_ratio, net_migration, apt_index])
    columns = ["지역명", "기준연월", "총인구수", "청년인구비율", "순이동인구수", "아파트매매가격지수"]
    return pd.DataFrame(rows, columns=columns)


# 3. 사이드바: 파일 업로드 및 자동 로드
st.sidebar.header("📂 데이터 및 분석 설정")
uploaded_files = st.sidebar.file_uploader("KOSIS 원본 CSV 3개(또는 통합 CSV 1개) 업로드", type=["csv"], accept_multiple_files=True)

before_clean_rows = 480
if uploaded_files:
    if len(uploaded_files) == 1:
        df = read_uploaded_csv(uploaded_files[0])
        before_clean_rows = len(df)
        df = df.dropna().reset_index(drop=True)
        st.sidebar.success(f"통합 CSV 로드 완료! ({len(df)}행)")
    else:
        try:
            df, before_clean_rows = merge_kosis_files(uploaded_files)
            st.sidebar.success(f"KOSIS 파일 3개 자동 병합 완료! ({len(df)}행)")
            st.sidebar.download_button(
                label="📥 병합된 merged_data.csv 다운로드",
                data=df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="merged_data.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.sidebar.error(f"병합 오류: {e}")
            df = generate_demo_data()
elif os.path.exists("merged_data.csv"):
    df = pd.read_csv("merged_data.csv")
    before_clean_rows = len(df)
    st.sidebar.success("저장소의 merged_data.csv를 자동으로 불러왔습니다!")
elif os.path.exists("data/merged_data.csv"):
    df = pd.read_csv("data/merged_data.csv")
    before_clean_rows = len(df)
    st.sidebar.success("저장소의 data/merged_data.csv를 자동으로 불러왔습니다!")
else:
    df = generate_demo_data()
    st.sidebar.info("현재 데모 데이터로 실행 중입니다. CSV 파일을 올리거나 저장소에 merged_data.csv를 추가하세요.")

selected_cities = st.sidebar.multiselect("분석할 지역 선택", options=df["지역명"].unique().tolist(), default=df["지역명"].unique().tolist())
filtered_df = df[df["지역명"].isin(selected_cities)].copy()

# 4. 상단 요약 지표 (KPI)
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 데이터 행 수", f"{len(filtered_df):,}행")
col2.metric("평균 청년인구비율", f"{filtered_df['청년인구비율'].mean():.2f}%")
col3.metric("평균 월 순이동인구", f"{filtered_df['순이동인구수'].mean():,.0f}명")
col4.metric("평균 아파트매매가격지수", f"{filtered_df['아파트매매가격지수'].mean():.2f}")

# 5. 회귀 모델 학습 및 평가
X = filtered_df[["청년인구비율", "순이동인구수"]]
y = filtered_df["아파트매매가격지수"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

linear_model = LinearRegression().fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)
poly_rmse = np.sqrt(mean_squared_error(y_test, poly_pred))
poly_r2 = r2_score(y_test, poly_pred)

# 6. 탭 구성
tab1, tab2, tab3, tab4 = st.tabs(["📋 데이터 개요·전처리", "📈 탐색적 데이터 분석(EDA)", "🤖 회귀 모델 비교·평가", "🔮 집값 예측 시뮬레이터"])

with tab1:
    st.subheader("1) 데이터 개요 (행·열 수, 열별 자료형 및 결측치)")
    st.write(f"- **전체 행 수 × 열 수:** `{len(df)}행 × {len(df.columns)}열` (선택된 지역 기준: `{len(filtered_df)}행`)")

    overview_df = pd.DataFrame({
        "열 이름": df.columns,
        "자료형(dtype)": df.dtypes.astype(str).values,
        "빈 값(결측치) 개수": df.isnull().sum().values,
        "예시 값": [str(df[c].iloc[0]) for c in df.columns]
    })
    st.table(overview_df)

    st.subheader("2) 결측치·이상치 처리 기준 및 처리 전후 행 수")
    st.markdown(f"""
    - **결측치 처리 기준:** 3개 통계표(주민등록인구·이동자수·매매가격지수)를 `지역명`과 `기준연월(2015.01~2024.12)` 기준으로 내부 병합(inner join)한 뒤, 수치 변환 불가(`NaN`) 행이 있으면 제거(`dropna`)했습니다.
    - **이상치 처리 기준:** 월별 순이동인구수의 급등락(신도시 입주 물량 집중 시기 등)은 실제 산업단지·택지개발 현상을 반영하는 핵심 데이터이므로 임의로 삭제하지 않고 원데이터 그대로 유지했습니다.
    - **처리 전후 행 수:** 병합 직후 **`{before_clean_rows}행`** → 결측치 제거 후 **`{len(df)}행`** (결측치 `0건`)
    """)

    st.subheader("3) 통합 데이터 미리보기")
    st.dataframe(filtered_df, use_container_width=True)

with tab2:
    st.subheader("청년인구비율 및 순이동인구수에 따른 아파트 매매가격지수 관계")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(
            filtered_df,
            x="청년인구비율",
            y="아파트매매가격지수",
            color="지역명",
            trendline="ols",
            title="청년인구비율(%) vs 아파트매매가격지수 (도시별 추세선)"
        )
        fig1.update_layout(xaxis_title="청년인구비율 (%)", yaxis_title="아파트매매가격지수")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.line(
            filtered_df,
            x="기준연월",
            y="아파트매매가격지수",
            color="지역명",
            title="연월별 아파트매매가격지수 추이 (2015.01~2024.12)"
        )
        fig2.update_layout(xaxis_title="기준연월", yaxis_title="아파트매매가격지수")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.box(
            filtered_df,
            x="지역명",
            y="청년인구비율",
            color="지역명",
            title="도시별 청년인구비율(%) 분포 비교 (상자그림)"
        )
        fig3.update_layout(xaxis_title="지역명", yaxis_title="청년인구비율 (%)")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.scatter(
            filtered_df,
            x="순이동인구수",
            y="아파트매매가격지수",
            color="지역명",
            trendline="ols",
            title="월 순이동인구수(명) vs 아파트매매가격지수"
        )
        fig4.update_layout(xaxis_title="월 순이동인구수 (명)", yaxis_title="아파트매매가격지수")
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("기준선(평균 모델) vs 회귀 모델 성능 비교")
    perf_df = pd.DataFrame({
        "모델명": ["기준선 (단순 평균 모델)", "다중 선형 회귀 (Linear)", "다항 회귀 2차 (Polynomial)"],
        "테스트 오차 (RMSE, 낮을수록 좋음)": [round(baseline_rmse, 3), round(linear_rmse, 3), round(poly_rmse, 3)],
        "결정계수 (R², 1에 가까울수록 좋음)": [0.0, round(linear_r2, 3), round(poly_r2, 3)]
    })
    st.table(perf_df)

with tab4:
    st.subheader("가상 조건을 입력해 아파트 매매가격지수 예측하기")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        input_youth = st.slider("청년인구비율 (%)", min_value=15.0, max_value=45.0, value=float(round(filtered_df["청년인구비율"].mean(), 1)), step=0.1)
    with sim_col2:
        input_migration = st.slider("월 순이동인구수 (명)", min_value=-2000, max_value=5000, value=int(filtered_df["순이동인구수"].mean()), step=50)

    input_df = pd.DataFrame([[input_youth, input_migration]], columns=["청년인구비율", "순이동인구수"])
    pred_value = linear_model.predict(input_df)[0]
    st.success(f"📊 예측된 **아파트 매매가격지수**는 **{pred_value:.2f}** 입니다.")
