import streamlit as st
import pandas as pd
import json
import os
from openai import OpenAI
from pypdf import PdfReader
import io

# ---------------------------------------------------------
# [설정] 페이지 기본 세팅
# ---------------------------------------------------------
st.set_page_config(page_title="DB Inc 프롬프팅 대회 채점기", layout="wide", page_icon="📊")

# Railway 환경변수 로드
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# [스타일] 차트 색상 등 설정
# ---------------------------------------------------------
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# [사이드바] 설정 및 파일 업로드
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 대회 설정")
    if api_key:
        st.success(f"✅ System Ready\n(GPT-5 nano)")
    else:
        st.error("❌ API Key Not Found")
        st.stop()
    
    st.divider()
    st.header("📂 데이터 파일")
    uploaded_context = st.file_uploader("1. 문맥 자료 (PDF/Txt/Excel)", type=['pdf', 'txt', 'xlsx'])
    uploaded_target = st.file_uploader("2. 정답지 (Txt/Excel)", type=['txt', 'xlsx'])
    uploaded_participants = st.file_uploader("3. 참가자 명단 (Excel)", type=['xlsx'])
    
    st.info("💡 심사평은 100자 이내로 요약되어 출력됩니다.")

# ---------------------------------------------------------
# [함수] 로직
# ---------------------------------------------------------
def read_file(file):
    if not file: return None
    ext = file.name.split('.')[-1].lower()
    try:
        if ext == 'pdf':
            reader = PdfReader(file)
            return "".join([page.extract_text() for page in reader.pages])
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(file).to_markdown(index=False)
        else:
            return file.getvalue().decode("utf-8")
    except:
        return ""

def evaluate(client, context, target, participants):
    results = []
    bar = st.progress(0)
    status = st.empty()
    total = len(participants)
    MODEL_NAME = "gpt-5-nano" 
    
    for idx, row in participants.iterrows():
        name = row.iloc[0]
        prompt = row.iloc[1]
        
        status.write(f"⚡ **{name}**님 채점 중... ({idx+1}/{total})")
        bar.progress((idx + 1) / total)
        
        try:
            # 1. 실행
            messages = [
                {"role": "system", "content": "데이터 분석 AI입니다."},
                {"role": "user", "content": f"---[Context]---\n{context}\n\n---[Prompt]---\n{prompt}"}
            ]
            
            # temperature 제거 (Default 사용)
            out1 = client.chat.completions.create(model=MODEL_NAME, messages=messages).choices[0].message.content
            out2 = client.chat.completions.create(model=MODEL_NAME, messages=messages).choices[0].message.content
            
            # 2. 심사 (100자 제한 적용)
            judge_prompt = f"""
            프롬프트 경진대회 심사위원입니다. 아래 기준에 따라 채점하세요.
            
            [평가 기준]
            1. 정확성(50점): 정답(Target)과 내용/형식 일치 여부
            2. 명확성(30점): 지시의 구체성과 논리성
            3. 재현성(20점): 2회 실행 결과의 동일성

            [데이터]
            - User Prompt: {prompt}
            - Target Answer: {target}
            - Output 1: {out1}
            - Output 2: {out2}
            
            JSON 포맷으로 응답하세요. 
            특히 'reasoning'(심사평)은 엑셀에 넣기 좋게 **반드시 100자 이내로 핵심만** 요약하세요.
            
            Format: {{ "accuracy": int, "clarity": int, "consistency": int, "reasoning": "100자 이내 요약(Korean)" }}
            """
            
            judge = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[{"role": "system", "content": "JSON output only."}, {"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"}
            )
            score_data = json.loads(judge.choices[0].message.content)
            
            total_score = score_data['accuracy'] + score_data['clarity'] + score_data['consistency']
            
            results.append({
                "순위": 0, 
                "이름": name,
                "총점": total_score,
                "정확성": score_data['accuracy'],
                "명확성": score_data['clarity'],
                "재현성": score_data['consistency'],
                "심사평": score_data['reasoning'], # 100자 제한됨
                "실행결과": out1
            })
            
        except Exception as e:
            results.append({
                "순위": 0, "이름": name, "총점": 0, 
                "정확성": 0, "명확성": 0, "재현성": 0,
                "심사평": "에러 발생", "실행결과": "Fail"
            })
            
    status.success("🎉 채점 완료!")
    bar.empty()
    return pd.DataFrame(results)

# ---------------------------------------------------------
# [메인] 대시보드 UI
# ---------------------------------------------------------
st.title("📊 DB Inc 프롬프팅 경진대회 대시보드")
st.markdown("### Powered by GPT-5 nano")

if st.button("🚀 채점 시작 (Start Grading)", type="primary", use_container_width=True):
    if not uploaded_context or not uploaded_target or not uploaded_participants:
        st.error("⚠️ 파일을 모두 업로드해주세요.")
    else:
        with st.spinner("데이터 분석 및 심사 진행 중..."):
            client = OpenAI(api_key=api_key)
            
            # 데이터 로드
            ctx = read_file(uploaded_context)
            tgt = read_file(uploaded_target)
            df_p = pd.read_excel(uploaded_participants)
            
            # 채점 실행
            res_df = evaluate(client, ctx, tgt, df_p)
            
            # 순위 정렬
            res_df = res_df.sort_values(by="총점", ascending=False).reset_index(drop=True)
            res_df["순위"] = res_df.index + 1
            
            # ==========================================
            # 1. 종합 지표 (KPI)
            # ==========================================
            st.divider()
            kpi1, kpi2, kpi3 = st.columns(3)
            
            avg_score = round(res_df['총점'].mean(), 1)
            max_score = res_df['총점'].max()
            winner_name = res_df.iloc[0]['이름']
            
            kpi1.metric("🏆 전체 참가자", f"{len(res_df)}명")
            kpi2.metric("📊 평균 점수", f"{avg_score}점")
            kpi3.metric("🥇 최고 점수", f"{max_score}점", f"1위: {winner_name}")
            
            # ==========================================
            # 2. 차트 시각화 (Visualization)
            # ==========================================
            st.divider()
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("📈 상위 10명 점수 현황")
                top_10 = res_df.head(10).sort_values('총점', ascending=True) # 차트는 아래부터 그려지므로 오름차순 정렬
                st.bar_chart(top_10.set_index("이름")["총점"], color="#FF4B4B", horizontal=True)

            with col_chart2:
                st.subheader("🧩 점수 구성 요소 분석 (Top 10)")
                # 정확성/명확성/재현성 누적 막대 그래프
                chart_data = top_10.set_index("이름")[["정확성", "명확성", "재현성"]]
                st.bar_chart(chart_data, horizontal=True)

            # ==========================================
            # 3. 리더보드 (Data Table)
            # ==========================================
            st.divider()
            st.subheader("📋 전체 리더보드")
            
            # 보기 좋게 컬럼 정리
            display_cols = ["순위", "이름", "총점", "정확성", "명확성", "재현성", "심사평"]
            
            # 스타일링된 데이터프레임 표시
            st.dataframe(
                res_df[display_cols],
                use_container_width=True,
                column_config={
                    "총점": st.column_config.ProgressColumn(
                        "총점", format="%d", min_value=0, max_value=100
                    ),
                    "심사평": st.column_config.TextColumn("심사평 (100자 요약)")
                },
                hide_index=True
            )
            
            # ==========================================
            # 4. 엑셀 다운로드
            # ==========================================
            output = io.BytesIO()
            
            # 엑셀 저장 시 실행결과까지 포함 (보기 편하게)
            save_cols = ["순위", "이름", "총점", "정확성", "명확성", "재현성", "심사평", "실행결과"]
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df[save_cols].to_excel(writer, index=False)
                
                # 엑셀 열 너비 자동 조정 (약간의 스타일링)
                worksheet = writer.sheets['Sheet1']
                worksheet.set_column('B:B', 15) # 이름
                worksheet.set_column('G:G', 50) # 심사평
                worksheet.set_column('H:H', 20) # 실행결과
            
            st.download_button(
                label="📥 결과 엑셀 다운로드 (Full Report)",
                data=output.getvalue(),
                file_name="DB_Inc_대회결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
