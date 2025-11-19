import streamlit as st
import pandas as pd
import json
import os
import asyncio
import time
from openai import AsyncOpenAI, RateLimitError
from pypdf import PdfReader
import io

# ---------------------------------------------------------
# [설정] 페이지 기본 세팅
# ---------------------------------------------------------
st.set_page_config(page_title="DB Inc 프롬프팅 대회 채점기", layout="wide", page_icon="⚡")

api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# [스타일]
# ---------------------------------------------------------
st.markdown("""
    <style>
    .metric-container { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .status-box { 
        padding: 15px; 
        border-radius: 8px; 
        margin-bottom: 10px; 
        text-align: center; 
        font-size: 1.1rem;
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        color: #1565c0;
        font-weight: bold;
    }
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
    
    # 모델 선택 (속도 핵심)
    st.subheader("🚀 모델 선택")
    model_name = st.selectbox(
        "사용할 모델", 
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4o-mini가 가장 빠르고 저렴합니다."
    )
    
    # 속도 설정
    st.subheader("⚡ 속도 설정")
    concurrency_limit = st.slider(
        "동시 채점 인원 (명)", 
        1, 20, 10, # 기본값을 10으로 상향
        help="gpt-4o-mini 기준 10명도 거뜬합니다."
    )
    
    st.divider()
    
    st.subheader("📂 데이터 업로드")
    uploaded_context = st.file_uploader("1. 문맥 자료", type=['pdf', 'txt', 'xlsx'])
    uploaded_target = st.file_uploader("2. 정답지", type=['txt', 'xlsx'])
    uploaded_participants = st.file_uploader("3. 참가자 명단", type=['xlsx'])
    
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
    """API 호출 실패 시 재시도"""
    for i in range(retries):
        try:
            return await client.chat.completions.create(model=model, messages=messages)
        except RateLimitError:
            await asyncio.sleep(1 + i) # 지수 백오프 아님, 짧게 대기
        except Exception:
            await asyncio.sleep(1) # 기타 에러 시 잠시 대기 후 재시도
    return None

async def evaluate_single_participant(sem, client, row, context, target, model_name, logs):
    name = row.iloc[0]
    prompt = row.iloc[1]
    
    async with sem: # [중요] 여기서 동시 실행 제어
        try:
            # 로그: 시작 알림
            logs.append(f"▶️ {name} 채점 시작...")
            
            # 1. 실행 (2회 동시 요청)
            messages = [
                {"role": "system", "content": "데이터 분석 AI입니다."},
                {"role": "user", "content": f"---[Context]---\n{context}\n\n---[Prompt]---\n{prompt}"}
            ]
            
            t1 = safe_api_call(client, model_name, messages)
            t2 = safe_api_call(client, model_name, messages)
            r1, r2 = await asyncio.gather(t1, t2)
            
            if not r1 or not r2: raise Exception("API 응답 없음")
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
            
            j_resp = await safe_api_call(client, model_name, 
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

async def run_all_evaluations(api_key, context, target, df_participants, limit, model_name):
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(limit) 
    tasks = []
    
    # UI 요소
    status_box = st.empty()
    progress_bar = st.progress(0)
    log_expander = st.expander("📜 실시간 채점 로그 (진행 상황)", expanded=True)
    log_placeholder = log_expander.empty()
    
    total_count = len(df_participants)
    start_time = time.time()
    logs = []
    
    # Task 생성
    for idx, row in df_participants.iterrows():
        task = evaluate_single_participant(sem, client, row, context, target, model_name, logs)
        tasks.append(task)
    
    results = []
    completed = 0
    
    # 실행 (as_completed로 완료되는 순서대로 처리)
    for f in asyncio.as_completed(tasks):
        result = await f
        results.append(result)
        completed += 1
        
        # 시간 계산
        elapsed = time.time() - start_time
        speed = completed / elapsed if elapsed > 0 else 0
        remaining = total_count - completed
        eta = int(remaining / speed) if speed > 0 else 0
        
        # UI 업데이트 (빈도 조절 없이 매번 업데이트하되, 내용은 심플하게)
        progress_bar.progress(completed / total_count)
        
        status_box.markdown(f"""
        <div class='status-box'>
        🚀 <b>{completed}</b> / {total_count} 명 완료
        <br><span style='font-size:0.9em'>⚡ 속도: 초당 {speed:.1f}명 처리 | ⏳ 남은 시간: 약 {eta}초</span>
        </div>
        """, unsafe_allow_html=True)
        
        # 로그 업데이트 (최신 5개만 표시)
        logs.append(f"✅ {result['이름']} 완료 ({result['총점']}점)")
        log_placeholder.text("\n".join(logs[-7:]))
        
    total_duration = time.time() - start_time
    return pd.DataFrame(results), total_duration

# ---------------------------------------------------------
# [메인] 실행 로직
# ---------------------------------------------------------
st.title("⚡ 초고속 채점 시스템 (Async + gpt-4o-mini)")

if st.button("🚀 채점 시작하기", type="primary", use_container_width=True):
    if not uploaded_context or not uploaded_target or not uploaded_participants:
        st.warning("⚠️ 파일 3개를 모두 업로드해주세요!")
    else:
        ctx = read_file(uploaded_context)
        tgt = read_file(uploaded_target)
        df_p = pd.read_excel(uploaded_participants)
        
        try:
            # 실행
            res_df, total_time = asyncio.run(run_all_evaluations(
                api_key, ctx, tgt, df_p, concurrency_limit, model_name
            ))
            
            # 결과 표시
            res_df = res_df.sort_values(by="총점", ascending=False).reset_index(drop=True)
            res_df["순위"] = res_df.index + 1
            
            st.success(f"🎉 채점 완료! (총 {int(total_time)}초 소요)")
            
            # 대시보드
            c1, c2, c3 = st.columns(3)
            c1.metric("참가자", f"{len(res_df)}명")
            c2.metric("1인당 평균 시간", f"{total_time/len(res_df):.2f}초")
            c3.metric("최고 점수", f"{res_df.iloc[0]['총점']}점")
            
            st.dataframe(res_df[["순위", "이름", "총점", "심사평"]], use_container_width=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
            st.download_button("💾 결과 엑셀 다운로드", output.getvalue(), "final_result.xlsx")
            
        except Exception as e:
            st.error(f"에러 발생: {e}")
