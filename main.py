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
st.set_page_config(page_title="DB Inc 프롬프팅 대회 채점기", layout="wide", page_icon="⚖️")

api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# [스타일] 대시보드 UI
# ---------------------------------------------------------
st.markdown("""
    <style>
    .metric-container { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    .status-box { 
        padding: 15px; border-radius: 8px; margin-bottom: 10px; text-align: center; 
        font-size: 1.1rem; background-color: #e8f5e9; border: 1px solid #c8e6c9; 
        color: #2e7d32; font-weight: bold;
    }
    .eta-text { color: #d32f2f; font-weight: bold; font-size: 0.9em; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [사이드바] 설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚖️ 최종 채점 시스템")
    if api_key:
        st.success(f"✅ API Key 연동 완료")
    else:
        st.error("❌ API Key 없음")
        st.stop()
    
    st.divider()
    
    st.info("ℹ️ GPT-5 nano 환경에 맞춰 Temperature=1.0(Default)이 강제 적용됩니다.")
    gen_model = "gpt-5-nano"  # 참가자 빙의용
    judge_model = "gpt-5-nano"     # 심사위원용
    
    concurrency_limit = st.slider("동시 채점 인원", 5, 50, 20, help="API 에러가 나면 줄이세요.")
    
    st.divider()
    st.subheader("📂 필수 파일 업로드")
    uploaded_context = st.file_uploader("1. 문맥 자료 (Input File)", type=['pdf', 'txt', 'xlsx', 'csv'])
    uploaded_target = st.file_uploader("2. 요구 산출물 (Target File)", type=['txt', 'xlsx', 'csv'])
    uploaded_participants = st.file_uploader("3. 참가자 명단 (Participants)", type=['xlsx'])
    
    if st.button("🧪 과제B 테스트용 샘플(20명) 받기"):
        dummy_data = {
            "이름": [f"참가자_{i+1:02d}" for i in range(20)],
            "프롬프트": [
                "데이터를 정제해. 1) id 중복제거 2) 'test' 포함된 유저 삭제 3) 날짜 YYYY-MM-DD 통일. 결과는 표로.",
                "데이터 정리해주고 테스트 유저도 포함해서 보여줘.",
                "날짜를 '24년 5월'로 바꾸고 Plan은 소문자로 해.",
                "그냥 알아서 잘 정리해봐.",
                "완벽하게 분석하고 [현상-원인-대책] 표로 정리해."
            ] * 4
        }
        df_dummy = pd.DataFrame(dummy_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_dummy.to_excel(writer, index=False)
        st.download_button("📥 샘플 엑셀 다운로드", output.getvalue(), "participants_sample.xlsx")

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
        elif ext == 'csv':
            return pd.read_csv(file).to_markdown(index=False)
        else:
            return file.getvalue().decode("utf-8")
    except:
        return ""

# [수정됨] kwargs를 받아 response_format 등을 처리할 수 있게 변경
async def safe_api_call(client, model, messages, retries=3, **kwargs):
    """API 호출 (RateLimit 처리 포함)"""
    for i in range(retries):
        try:
            return await client.chat.completions.create(model=model, messages=messages, **kwargs)
        except RateLimitError:
            await asyncio.sleep(1 + (i * 0.5))
        except Exception as e:
            print(f"API Error: {e}")  # 디버깅용 로그
            await asyncio.sleep(0.5)
    return None

async def generate_output(client, model, context, prompt):
    messages = [
        {"role": "system", "content": "당신은 데이터 처리 엔진입니다. 지시에 따라 결과를 출력하세요."},
        {"role": "user", "content": f"---[Input File]---\n{context}\n\n---[User Prompt]---\n{prompt}"}
    ]
    t1 = safe_api_call(client, model, messages)
    t2 = safe_api_call(client, model, messages)
    r1, r2 = await asyncio.gather(t1, t2)
    
    out1 = r1.choices[0].message.content if r1 else "Error"
    out2 = r2.choices[0].message.content if r2 else "Error"
    return out1, out2

# [수정됨] JSON 파싱 오류 해결을 위한 강력한 처리 로직
async def audit_submission(client, model, target, out1, out2, original_prompt):
    judge_prompt = f"""
    당신은 프롬프트 경진대회의 **엄격한 심사위원**입니다.
    아래 [평가 기준]에 따라 점수를 매기고 JSON 형식으로만 출력하세요.

    [평가 기준 - 총 100점]
    1. Accuracy (50점): Target과 실행 결과(Value, Format) 일치 여부.
    2. Clarity (30점): 프롬프트의 명확성.
    3. Consistency (20점): 실행 결과 1과 2의 동일성.

    [입력 데이터]
    - 참가자 프롬프트: "{original_prompt}"
    - Target: {target[:2000]}
    - Result 1: {out1[:2000]}
    - Result 2: {out2[:2000]}

    [필수 출력 형식 (JSON)]
    {{
        "accuracy": int, 
        "clarity": int, 
        "consistency": int, 
        "feedback": "String (200자 이내)"
    }}
    """
    
    # JSON 모드 활성화 및 호출
    resp = await safe_api_call(
        client, 
        model, 
        [{"role": "system", "content": "You are a judge. Output JSON only."}, {"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"} # 중요: 강제 JSON 모드
    )
    
    if not resp:
        return {"accuracy": 0, "clarity": 0, "consistency": 0, "feedback": "API 호출 실패"}

    try:
        content = resp.choices[0].message.content
        # 마크다운 코드 블록 제거 (혹시 모를 경우 대비)
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except json.JSONDecodeError:
        return {"accuracy": 0, "clarity": 0, "consistency": 0, "feedback": "JSON 파싱 실패"}
    except Exception as e:
        return {"accuracy": 0, "clarity": 0, "consistency": 0, "feedback": f"알 수 없는 에러: {str(e)}"}

async def process_participant(sem, client, row, context, target, gen_model, judge_model):
    name = row.iloc[0]
    prompt = row.iloc[1]
    
    async with sem:
        try:
            out1, out2 = await generate_output(client, gen_model, context, prompt)
            score_data = await audit_submission(client, judge_model, target, out1, out2, prompt)
            
            total = score_data.get('accuracy', 0) + score_data.get('clarity', 0) + score_data.get('consistency', 0)
            
            return {
                "이름": name, 
                "총점": total,
                "정확성": score_data.get('accuracy', 0), 
                "명확성": score_data.get('clarity', 0), 
                "재현성": score_data.get('consistency', 0),
                "피드백": score_data.get('feedback', "피드백 없음"),
                "결과물": out1[:100] + "..."
            }
        except Exception as e:
            return {"이름": name, "총점": 0, "피드백": f"System Error: {e}", "결과물": "Fail"}

async def run_grading_system(api_key, context, target, df_p, limit, gen_model, judge_model):
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(limit)
    tasks = []
    
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    total_count = len(df_p)
    start_time = time.time()
    
    for idx, row in df_p.iterrows():
        tasks.append(process_participant(sem, client, row, context, target, gen_model, judge_model))
    
    results = []
    completed = 0
    
    for f in asyncio.as_completed(tasks):
        res = await f
        results.append(res)
        completed += 1
        
        elapsed = time.time() - start_time
        avg_speed = elapsed / completed if completed > 0 else 0
        remaining = total_count - completed
        eta_seconds = int(avg_speed * remaining)
        eta_str = f"{eta_seconds // 60}분 {eta_seconds % 60}초" if eta_seconds >= 60 else f"{eta_seconds}초"
        
        progress_bar.progress(completed / total_count)
        status_box.markdown(f"""
        <div class='status-box'>
        🚀 채점 진행 중... ({completed} / {total_count}명)<br>
        <span class='eta-text'>⚡ 속도: {avg_speed:.1f}초/명 | ⏳ 남은 시간: {eta_str}</span>
        </div>
        """, unsafe_allow_html=True)
        
    return pd.DataFrame(results), time.time() - start_time

# ---------------------------------------------------------
# [메인] 실행
# ---------------------------------------------------------
st.title("📏 DB Inc 프롬프팅 경진대회 채점 시스템")

if st.button("🚀 정밀 채점 시작 (Strict Mode)", type="primary", use_container_width=True):
    if not uploaded_context or not uploaded_target or not uploaded_participants:
        st.warning("⚠️ 파일 3개를 모두 업로드해주세요.")
    else:
        ctx = read_file(uploaded_context)
        tgt = read_file(uploaded_target)
        df_p = pd.read_excel(uploaded_participants)
        
        try:
            # Windows/Streamlit 환경 호환성을 위한 loop 정책 설정
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # 이미 루프가 돌고 있을 경우를 대비해 run 대신 loop 활용 고려 (Streamlit 특성상 run 사용하되 에러 핸들링)
            # 여기서는 가장 심플하게 run 사용
            res_df, total_time = asyncio.run(run_grading_system(
                api_key, ctx, tgt, df_p, concurrency_limit, gen_model, judge_model
            ))
            
            # 데이터 타입 정리 (차트 에러 방지)
            res_df["총점"] = pd.to_numeric(res_df["총점"], errors='coerce').fillna(0)
            res_df = res_df.sort_values(by="총점", ascending=False).reset_index(drop=True)
            res_df["순위"] = res_df.index + 1
            
            st.success(f"✅ 채점 완료! (총 소요 시간: {int(total_time)}초)")
            
            st.markdown("### 📊 결과 대시보드")
            k1, k2, k3 = st.columns(3)
            k1.metric("참가자", f"{len(res_df)}명")
            k2.metric("평균 점수", f"{round(res_df['총점'].mean(), 1)}점")
            if not res_df.empty:
                k3.metric("최고 점수", f"{res_df.iloc[0]['총점']}점", res_df.iloc[0]['이름'])
            
            st.divider()
            st.caption("상위권 점수 분포")
            # 차트 데이터 유효성 검사 후 그리기
            if not res_df.empty:
                chart_data = res_df.head(15).set_index("이름")[["총점"]]
                st.bar_chart(chart_data, color="#2e7d32")
            
            st.divider()
            st.subheader("📋 상세 성적표")
            st.dataframe(
                res_df[["순위", "이름", "총점", "정확성", "명확성", "재현성", "피드백"]],
                use_container_width=True,
                column_config={
                    "총점": st.column_config.ProgressColumn("총점", min_value=0, max_value=100, format="%d점"),
                    "피드백": st.column_config.TextColumn("심사평", width="large")
                }
            )
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
                worksheet = writer.sheets['Sheet1']
                workbook = writer.book
                format_wrap = workbook.add_format({'text_wrap': True})
                worksheet.set_column('G:G', 60, format_wrap)
                
            st.download_button("💾 최종 결과 엑셀 다운로드", output.getvalue(), "final_result.xlsx", type="primary")
            
        except Exception as e:
            st.error(f"시스템 에러 발생: {e}")
            st.error("TIP: API 키를 확인하거나, 동시 채점 인원(Concurrency)을 줄여보세요.")
