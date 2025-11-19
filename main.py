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
    
    # 모델 설정 (GPT-5 nano 대응을 위해 gpt-4o-mini 사용하되 temp=1 강제)
    st.info("ℹ️ GPT-5 nano(gpt-4o-mini) 환경에 맞춰 Temperature=1.0(Default)이 강제 적용됩니다.")
    gen_model = "gpt-4o-mini"  # 참가자 빙의용
    judge_model = "gpt-4o"     # 심사위원용 (더 똑똑한 모델 권장)
    
    # 속도 조절
    concurrency_limit = st.slider("동시 채점 인원", 10, 50, 30, help="API 에러가 나면 줄이세요.")
    
    st.divider()
    st.subheader("📂 필수 파일 업로드")
    uploaded_context = st.file_uploader("1. 문맥 자료 (Input File)", type=['pdf', 'txt', 'xlsx', 'csv'])
    uploaded_target = st.file_uploader("2. 요구 산출물 (Target File)", type=['txt', 'xlsx', 'csv'])
    uploaded_participants = st.file_uploader("3. 참가자 명단 (Participants)", type=['xlsx'])
    
    if st.button("🧪 과제B 테스트용 샘플(20명) 받기"):
        dummy_data = {
            "이름": [f"참가자_{i+1:02d}" for i in range(20)],
            "프롬프트": [
                # 1. 정답 (100점)
                "데이터를 정제해. 1) id 중복제거 2) 'test' 포함된 유저 삭제 3) 날짜 YYYY-MM-DD 통일 4) Plan 대문자 변환. 결과는 표로 출력해.",
                # 2. 오답 (테스트 유저 포함 - 감점)
                "데이터 정리해주고 테스트 유저도 포함해서 보여줘.",
                # 3. 오답 (형식 틀림)
                "날짜를 '24년 5월'로 바꾸고 Plan은 소문자로 해.",
                # 4. 모호함 (감점)
                "그냥 알아서 잘 정리해봐.",
                # 5. 짝수/홀수 섞기
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

async def safe_api_call(client, model, messages, retries=3):
    """API 호출 (Temperature 파라미터 제거 - 400 에러 방지)"""
    for i in range(retries):
        try:
            # Temperature 설정 제거 -> 기본값(1) 사용
            return await client.chat.completions.create(model=model, messages=messages)
        except RateLimitError:
            await asyncio.sleep(1 + (i * 0.5))
        except Exception:
            await asyncio.sleep(0.5)
    return None

# 1. 참가자 프롬프트 실행 (데이터 생성)
async def generate_output(client, model, context, prompt):
    messages = [
        {"role": "system", "content": "당신은 데이터 처리 엔진입니다. 사용자의 지시에 따라 데이터를 처리하고 결과(표/텍스트)만 출력하세요. 사족은 붙이지 마세요."},
        {"role": "user", "content": f"---[Input File]---\n{context}\n\n---[User Prompt]---\n{prompt}"}
    ]
    # 동시성 체크를 위해 2번 실행 (Consistency 평가용)
    t1 = safe_api_call(client, model, messages)
    t2 = safe_api_call(client, model, messages)
    r1, r2 = await asyncio.gather(t1, t2)
    
    out1 = r1.choices[0].message.content if r1 else "Error"
    out2 = r2.choices[0].message.content if r2 else "Error"
    return out1, out2

# 2. 심사 (Auditor)
async def audit_submission(client, model, target, out1, out2, original_prompt):
    # 심사 기준 (엄격 준수)
    judge_prompt = f"""
    당신은 프롬프트 경진대회의 **엄격한 심사위원**입니다.
    아래 [평가 기준]에 따라 깐깐하게 점수를 매기세요.
    
    [평가 기준 - 총 100점]
    
    1. **정확성 (Accuracy) - 50점**
       - 참가자의 [실행 결과]가 [요구 산출물(Target)]과 데이터 값, 형식이 일치하는가?
       - **핵심 감점 요인:** - 정답에 없는 데이터(예: 삭제했어야 할 Test 유저)가 남아있으면 **-20점**.
         - 날짜 포맷, 대소문자 규정이 틀리면 **-10점**.
       - 50점: 완벽 일치 / 30점: 일부 차이 / 20점 이하: 불일치.

    2. **명확성 (Clarity) - 30점**
       - [참가자 프롬프트]가 명확한 역할(페르소나)과 단계별 지시를 포함하는가?
       - "알아서 해줘", "요약해" 처럼 모호하면 **10점 이하**.
       - 구체적 조건(포맷, 제외조건 등)이 명시되었으면 **30점**.

    3. **규칙 및 검증 (Consistency) - 20점**
       - [실행 결과 1]과 [실행 결과 2]가 동일한가?
       - 실행할 때마다 결과가 달라지면 재현성이 없는 것으로 간주하여 **10점 이하**.

    [비교할 데이터]
    - 참가자 프롬프트: "{original_prompt}"
    - 요구 산출물 (Target): {target[:3000]}... (생략됨)
    - 실행 결과 1: {out1[:3000]}...
    - 실행 결과 2: {out2[:3000]}...

    [출력 형식 (JSON Only)]
    {{
        "accuracy": int, 
        "clarity": int, 
        "consistency": int, 
        "feedback": "참가자를 위한 피드백 (200자 이내로 요약할 것)"
    }}
    """
    
    resp = await safe_api_call(client, model, 
        [{"role": "system", "content": "JSON output only."}, {"role": "user", "content": judge_prompt}])
    
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"accuracy": 0, "clarity": 0, "consistency": 0, "feedback": "JSON 파싱 에러"}

# 3. 프로세스 통합
async def process_participant(sem, client, row, context, target, gen_model, judge_model):
    name = row.iloc[0]
    prompt = row.iloc[1]
    
    async with sem:
        try:
            # 1단계: 생성 (Generation)
            out1, out2 = await generate_output(client, gen_model, context, prompt)
            
            # 2단계: 심사 (Auditing)
            # 프롬프트는 전체를 다 넘기되, 결과 리턴값인 feedback만 200자로 제한됨
            score_data = await audit_submission(client, judge_model, target, out1, out2, prompt)
            
            total = score_data['accuracy'] + score_data['clarity'] + score_data['consistency']
            
            return {
                "이름": name, "총점": total,
                "정확성": score_data['accuracy'], "명확성": score_data['clarity'], "재현성": score_data['consistency'],
                "피드백": score_data['feedback'], # 200자 요약됨
                "결과물": out1[:100] + "..." # 미리보기
            }
        except Exception as e:
            return {"이름": name, "총점": 0, "피드백": f"Error: {e}", "결과물": "Fail"}

async def run_grading_system(api_key, context, target, df_p, limit, gen_model, judge_model):
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(limit)
    tasks = []
    
    # UI 상태창
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
        
        # ETA 계산
        elapsed = time.time() - start_time
        avg_speed = elapsed / completed
        remaining = total_count - completed
        eta_seconds = int(avg_speed * remaining)
        
        if eta_seconds >= 60:
            eta_str = f"{eta_seconds // 60}분 {eta_seconds % 60}초"
        else:
            eta_str = f"{eta_seconds}초"
            
        progress_bar.progress(completed / total_count)
        
        status_box.markdown(f"""
        <div class='status-box'>
        🚀 채점 진행 중... ({completed} / {total_count}명)<br>
        <span class='eta-text'>⚡ 평균 속도: {avg_speed:.1f}초/명 | ⏳ 예상 남은 시간: 약 {eta_str}</span>
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
            res_df, total_time = asyncio.run(run_grading_system(
                api_key, ctx, tgt, df_p, concurrency_limit, gen_model, judge_model
            ))
            
            res_df = res_df.sort_values(by="총점", ascending=False).reset_index(drop=True)
            res_df["순위"] = res_df.index + 1
            
            st.success(f"✅ 채점 완료! (총 소요 시간: {int(total_time)}초)")
            
            # 1. 대시보드 KPI
            st.markdown("### 📊 결과 대시보드")
            k1, k2, k3 = st.columns(3)
            k1.metric("참가자", f"{len(res_df)}명")
            k2.metric("평균 점수", f"{round(res_df['총점'].mean(), 1)}점")
            k3.metric("최고 점수", f"{res_df.iloc[0]['총점']}점", res_df.iloc[0]['이름'])
            
            # 2. 차트
            st.divider()
            st.caption("상위권 점수 분포")
            st.bar_chart(res_df.head(15).set_index("이름")["총점"], color="#2e7d32")
            
            # 3. 상세 리스트
            st.divider()
            st.subheader("📋 상세 성적표")
            st.dataframe(
                res_df[["순위", "이름", "총점", "정확성", "명확성", "재현성", "피드백"]],
                use_container_width=True,
                column_config={
                    "총점": st.column_config.ProgressColumn("총점", min_value=0, max_value=100, format="%d점"),
                    "피드백": st.column_config.TextColumn("심사평 (200자 요약)", width="large")
                }
            )
            
            # 4. 엑셀 다운로드
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
                # 서식 적용
                workbook = writer.book
                worksheet = writer.sheets['Sheet1']
                format_wrap = workbook.add_format({'text_wrap': True})
                worksheet.set_column('G:G', 60, format_wrap) # 피드백 컬럼 넓게
                
            st.download_button("💾 최종 결과 엑셀 다운로드", output.getvalue(), "final_grading_result.xlsx", type="primary")
            
        except Exception as e:
            st.error(f"에러 발생: {e}")
