import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

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

# 2. 데이터 로드 함수 (실제 CSV가 있으면 불러오고, 없으면 구조가 동일한 예시 데이터 생성)
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df, False
    else:
        # 480행 × 6열 구조의 데모용 데이터 생성 (실제 파일 업로드 전 테스트용)
        np.random.seed(42)
        cities = ["평택시", "용인시", "화성시", "이천시"]
        dates = pd.date_range("2015-01-01", "2024-12-01", freq="MS").strftime("%Y.%m")
        rows = []
        for city in cities:
            base_pop = {"평택시": 450000, "용인시": 960000, "화성시": 540000, "이천시": 205000}[city]
            base_youth = {"평택시": 28.5, "용인시": 27.0, "화성시": 31.0, "이천시": 27.5}[city]
            for i, d in enumerate(dates):
                total_pop = int(base_pop + i * np.random.uniform(800, 1800))
                youth_ratio = round(base_youth + (i * 0.015) + np.random.normal(0, 0.3), 2)
                net_migration = int(np.random.normal(800, 450))
                # 회귀 관계가 드러나도록 매매가격지수 생성
                apt_index = round(
                    65.0 + (youth_ratio - 26.0) * 4.2 + (net_migration * 0.004) + (i * 0.18) + np.random.normal(0, 2.5),
                    2
                )
                rows.append([city, d, total_pop, youth_ratio, net_migration, apt_index])
        columns = ["지역명", "기준연월", "총인구수", "청년인구비율", "순이동인구수", "아파트매매가격지수"]
        return pd.DataFrame(rows, columns=columns), True

# 3. 사이드바: CSV 파일 업로드 및 지역 필터
st.sidebar.header("📂 데이터 및 분석 설정")
uploaded_file = st.sidebar.file_uploader("병합한 CSV 파일 업로드 (선택)", type=["csv"])
df, is_demo = load_data(uploaded_file)

if is_demo:
    st.sidebar.info("현재 **데모 데이터(480행)**로 실행 중입니다. 전처리한 실제 CSV 파일을 올리면 즉시 반영됩니다.")
else:
    st.sidebar.success("업로드한 실제 데이터로 분석 중입니다!")

selected_cities = st.sidebar.multiselect(
    "분석할 지역 선택",
    options=df["지역명"].unique().tolist(),
    default=df["지역명"].unique().tolist()
)

filtered_df = df[df["지역명"].isin(selected_cities)].copy()

# 4. 상단 요약 지표 (KPI)
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 데이터 행 수", f"{len(filtered_df):,}행")
col2.metric("평균 청년인구비율", f"{filtered_df['청년인구비율'].mean():.2f}%")
col3.metric("평균 월 순이동인구", f"{filtered_df['순이동인구수'].mean():,.0f}명")
col4.metric("평균 아파트매매가격지수", f"{filtered_df['아파트매매가격지수'].mean():.2f}")

# 5. 회귀 모델 학습 및 평가 (80% 훈련 / 20% 테스트)
X = filtered_df[["청년인구비율", "순이동인구수"]]
y = filtered_df["아파트매매가격지수"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 기준선 (평균 모델)
baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

# 후보 모델 1: 다중 선형 회귀
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

# 후보 모델 2: 다항 회귀 (2차)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)
poly_rmse = np.sqrt(mean_squared_error(y_test, poly_pred))
poly_r2 = r2_score(y_test, poly_pred)

# 6. 탭 구성: 시각화 / 모델 성능 비교 / 직접 예측 시뮬레이터
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 데이터 시각화",
    "🤖 회귀 모델 비교·평가",
    "🔮 집값 예측 시뮬레이터",
    "📋 원본 데이터 확인"
])

with tab1:
    st.subheader("청년인구비율 및 순이동인구수에 따른 아파트 매매가격지수 관계")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(
            filtered_df,
            x="청년인구비율",
            y="아파트매매가격지수",
            color="지역명",
            trendline="ols",
            title="청년인구비율(%) vs 아파트매매가격지수"
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.line(
            filtered_df,
            x="기준연월",
            y="아파트매매가격지수",
            color="지역명",
            title="연월별 아파트매매가격지수 추이 (2015~2024)"
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("기준선(평균 모델) vs 회귀 모델 성능 비교")
    perf_df = pd.DataFrame({
        "모델명": ["기준선 (단순 평균 모델)", "다중 선형 회귀 (Linear)", "다항 회귀 2차 (Polynomial)"],
        "테스트 오차 (RMSE, 낮을수록 좋음)": [round(baseline_rmse, 3), round(linear_rmse, 3), round(poly_rmse, 3)],
        "결정계수 (R², 1에 가까울수록 좋음)": [0.0, round(linear_r2, 3), round(poly_r2, 3)]
    })
    st.table(perf_df)

    st.markdown(f"""
    **해석 요약 (다중 선형 회귀 기준):**
    - **청년인구비율 회귀계수:** `{linear_model.coef_[0]:.3f}` *(청년 비율이 1%p 오를 때 매매가격지수 변화량)*
    - **순이동인구수 회귀계수:** `{linear_model.coef_[1]:.5f}` *(순이동이 1명 늘 때 매매가격지수 변화량)*
    - 기준선 오차(`{baseline_rmse:.2f}`) 대비 회귀 모델의 오차(`{linear_rmse:.2f}`)가 크게 줄어들어 예측 설명력이 확인됩니다.
    """)

with tab3:
    st.subheader("가상 조건을 입력해 아파트 매매가격지수 예측하기")
    st.write("슬라이더를 움직여 청년 인구 비율과 순이동 인구수가 변할 때 예측 매매가격지수가 어떻게 달라지는지 확인해 보세요.")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        input_youth = st.slider(
            "청년인구비율 (%)",
            min_value=15.0,
            max_value=45.0,
            value=float(round(filtered_df["청년인구비율"].mean(), 1)),
            step=0.1
        )
    with sim_col2:
        input_migration = st.slider(
            "월 순이동인구수 (명)",
            min_value=-2000,
            max_value=5000,
            value=int(filtered_df["순이동인구수"].mean()),
            step=50
        )

    pred_value = linear_model.predict([[input_youth, input_migration]])[0]
    st.success(f"📊 예측된 **아파트 매매가격지수**는 **{pred_value:.2f}** 입니다.")

with tab4:
    st.subheader("통합 데이터 미리보기 (480행 × 6열 구조)")
    st.dataframe(filtered_df, use_container_width=True)
