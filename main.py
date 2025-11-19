import streamlit as st
import pandas as pd
import json
import os
import asyncio
import time  # 시간 측정을 위해 추가
from openai import AsyncOpenAI, RateLimitError
from pypdf import PdfReader
import io

# ---------------------------------------------------------
# [설정] 페이지 기본 세팅
# ---------------------------------------------------------
st.set_page_config(page_title="DB Inc 프롬프팅 대회 채점기", layout="wide", page_icon="⏱️")

api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# [스타일]
# ---------------------------------------------------------
st.markdown("""
    <style>
    .metric-container { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .status-box { 
        background-color: #e8f4f8; 
        padding: 15px; 
        border-radius: 8px; 
        margin-bottom: 20px; 
        text-align: center; 
        font-size: 1.1rem;
        border: 1px solid #b3e5fc;
    }
    .time-highlight { color: #0068c9; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [사이드바]
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    
    if api_key:
        st.success(f"✅ API Key 연동 완료")
    else:
        st.error("❌ API Key 없음")
        st.stop()
    
    st.divider()
    
    # 속도 설정
    st.subheader("⚡ 속도 설정")
    concurrency_limit = st.slider(
        "동시 채점 인원 (명)", 
        1, 10, 5,
        help="5명 설정을 권장합니다. (40명 기준 약 3~4분 소요)"
    )
    
    st.divider()
    
    st.subheader("📂 데이터 업로드")
    uploaded_context = st.file_uploader("1. 문맥 자료", type=['pdf', 'txt', 'xlsx'])
    uploaded_target = st.file_uploader("2. 정답지", type=['txt', 'xlsx'])
    uploaded_participants = st.file_uploader("3. 참가자 명단", type=['xlsx'])
    
    st.divider()
    
    # 더미 데이터 생성
    if st.button("🧪 테스트용 샘플(20명) 다운로드"):
        dummy_data = {
            "이름": [f"참가자_{i+1:02d}" for i in range(20)],
            "프롬프트": [
                "데이터를 분석해서 요약해줘." if i % 2 == 0 else "상세하게 분석하고 표로 그려줘."
                for i in range(20)
            ]
        }
        df_dummy = pd.DataFrame(dummy_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_dummy.to_excel(writer, index=False)
        st.download_button("📥 샘플 엑셀(20명) 받기", output.getvalue(), "participants_20.xlsx")

# ---------------------------------------------------------
# [함수] 로직
# ---------------------------------------------------------
def read_file(file):
    if not file: return ""
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

async def safe_api_call(client, model, messages, retries=3):
    for i in range(retries):
        try:
            return await client.chat.completions.create(model=model, messages=messages)
        except RateLimitError:
            await asyncio.sleep((i + 1) * 2)
        except Exception as e:
            raise e
    return None

async def evaluate_single_participant(sem, client, row, context, target):
    name = row.iloc[0]
    prompt = row.iloc[1]
    MODEL_NAME = "gpt-5-nano"
    
    async with sem:
        try:
            # 1. 실행
            messages = [
                {"role": "system", "content": "데이터 분석 AI입니다."},
                {"role": "user", "content": f"---[Context]---\n{context}\n\n---[Prompt]---\n{prompt}"}
            ]
            
            t1 = safe_api_call(client, MODEL_NAME, messages)
            t2 = safe_api_call(client, MODEL_NAME, messages)
            r1, r2 = await asyncio.gather(t1, t2)
            
            if not r1 or not r2: raise Exception("API Error")
            out1 = r1.choices[0].message.content
            out2 = r2.choices[0].message.content
            
            # 2. 심사
            judge_prompt = f"""
            [평가 기준]
            1. 정확성(50): 정답 일치 여부
            2. 명확성(30): 지시 구체성
            3. 재현성(20): 결과 동일성

            [Data]
            - Prompt: {prompt}
            - Target: {target}
            - Out1: {out1}
            - Out2: {out2}
            
            Return JSON: {{ "accuracy": int, "clarity": int, "consistency": int, "reasoning": "200자 이내 요약(Korean)" }}
            """
            
            j_resp = await safe_api_call(client, MODEL_NAME, 
                [{"role": "system", "content": "JSON output only."}, {"role": "user", "content": judge_prompt}])
            
            score = json.loads(j_resp.choices[0].message.content)
            total = score['accuracy'] + score['clarity'] + score['consistency']
            
            return {
                "이름": name, "총점": total,
                "정확성": score['accuracy'], "명확성": score['clarity'], "재현성": score['consistency'],
                "심사평": score['reasoning'], "실행결과": out1
            }
        except Exception as e:
            return { "이름": name, "총점": 0, "심사평": f"Error: {e}", "실행결과": "Fail" }

async def run_all_evaluations(api_key, context, target, df_participants, limit):
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(limit) 
    tasks = []
    
    # UI 요소
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    total_count = len(df_participants)
    start_time = time.time() # 전체 시작 시간
    
    # Task 생성
    for idx, row in df_participants.iterrows():
        task = evaluate_single_participant(sem, client, row, context, target)
        tasks.append(task)
    
    results = []
    completed = 0
    
    # 실행 및 실시간 시간 계산
    for f in asyncio.as_completed(tasks):
        result = await f
        results.append(result)
        completed += 1
        
        # 시간 계산
        elapsed_time = time.time() - start_time
        avg_time_per_person = elapsed_time / completed
        remaining_people = total_count - completed
        eta_seconds = int(avg_time_per_person * remaining_people)
        
        # ETA 포맷팅 (분:초)
        if eta_seconds >= 60:
            eta_str = f"{eta_seconds // 60}분 {eta_seconds % 60}초"
        else:
            eta_str = f"{eta_seconds}초"
            
        # UI 업데이트 (진행률 + 남은 시간)
        progress = completed / total_count
        progress_bar.progress(progress)
        
        status_box.markdown(f"""
        <div class="status-box">
            🔄 <b>채점 진행 중...</b> ({completed} / {total_count}명)<br>
            <span style="font-size: 0.9em; color: #555;">
            ⏱️ 경과 시간: {int(elapsed_time)}초 | ⏳ <b>예상 남은 시간: {eta_str}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    total_duration = time.time() - start_time
    status_box.success(f"✅ 채점 완료! (총 소요 시간: {int(total_duration)}초)")
    
    return pd.DataFrame(results), total_duration

# ---------------------------------------------------------
# [메인] 실행 로직
# ---------------------------------------------------------
st.title("⏱️ DB Inc 프롬프팅 대회 (Time Tracker)")

if st.button("🚀 채점 시작하기", type="primary", use_container_width=True):
    if not uploaded_context or not uploaded_target or not uploaded_participants:
        st.warning("⚠️ 파일 3개를 모두 업로드해주세요!")
    else:
        ctx = read_file(uploaded_context)
        tgt = read_file(uploaded_target)
        df_p = pd.read_excel(uploaded_participants)
        
        try:
            # 실행 (결과 DF + 총 시간 반환)
            res_df, total_time = asyncio.run(run_all_evaluations(api_key, ctx, tgt, df_p, concurrency_limit))
            
            # 결과 정렬
            res_df = res_df.sort_values(by="총점", ascending=False).reset_index(drop=True)
            res_df["순위"] = res_df.index + 1
            
            # -------------------------------------------------
            # 📊 대시보드 섹션
            # -------------------------------------------------
            st.divider()
            st.markdown("### ⏱️ 소요 시간 분석")
            
            t1, t2, t3 = st.columns(3)
            avg_time = total_time / len(res_df)
            
            t1.metric("🕒 총 소요 시간", f"{int(total_time)}초", help="전체 채점에 걸린 실제 시간입니다.")
            t2.metric("⚡ 1인당 평균 속도", f"{round(avg_time, 2)}초", help="참가자 한 명을 채점하는 데 걸린 평균 시간입니다.")
            t3.metric("🚀 처리 효율 (TPM)", f"{round(60/avg_time * concurrency_limit, 1)}건", help="분당 처리 가능한 예상 건수입니다.")
            
            st.divider()
            st.markdown("### 🏆 채점 결과")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("👥 참가자", f"{len(res_df)}명")
            k2.metric("📈 평균 점수", f"{round(res_df['총점'].mean(), 1)}점")
            k3.metric("🥇 1위", res_df.iloc[0]['이름'], f"{res_df.iloc[0]['총점']}점")
            
            st.bar_chart(res_df.head(10).set_index("이름")["총점"], color="#00CC96")
            
            st.dataframe(res_df[["순위", "이름", "총점", "심사평"]], use_container_width=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("💾 결과 엑셀 다운로드", output.getvalue(), "result_time_tracked.xlsx", type="primary")
            
        except Exception as e:
            st.error(f"에러 발생: {e}")
