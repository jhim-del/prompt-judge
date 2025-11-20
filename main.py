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
st.set_page_config(page_title="DB Inc 프롬프팅 대회 채점기 v2.0", layout="wide", page_icon="⚖️")

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
    st.header("채점 시스템 v2.0 (Improved)")

    # 환경 변수에 키가 없으면 입력창 표시
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("🔑 OpenAI API Key 입력", type="password", help="sk-...로 시작하는 키를 입력하세요.")

    if api_key:
        st.success(f"✅ API Key 확인됨")
    else:
        st.warning("⚠️ API Key를 입력해주세요")
        st.stop() 
    
    st.divider()
    
    st.info("ℹ️ 개선된 로직: 정확성(40) + 일반화(40) + 명확성(20)")
    
    # 모델 설정
    gen_model = "gpt-4o-mini"  # 참가자 빙의용 (결과 생성)
    judge_model = "gpt-4o-mini" # 심사위원용 (채점)
    
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
                "데이터를 정제해. 1) id 중복제거 2) 'test' 포함된 유저 삭제 3) 날짜 YYYY-MM-DD 통일. 결과는 표로.", # 좋은 예시
                "데이터 정리해주고 테스트 유저도 포함해서 보여줘.", # 모호함
                "날짜를 '24년 5월'로 바꾸고 Plan은 소문자로 해.", # 부분적
                "그냥 알아서 잘 정리해봐.", # 나쁜 예시
                "너는 데이터 분석 전문가야. 다음 규칙을 엄격히 준수해.\n[규칙]\n1. ID 중복 제거\n2. 예외 처리: 날짜 형식이 다르면 'N/A'로 표기" # 일반화된 예시
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
            # sheet_name=None이면 모든 시트를 dict 형태로 가져옴 {'Sheet1': df, 'Sheet2': df...}
            sheets_dict = pd.read_excel(file, sheet_name=None)
            all_text = []
            for sheet_name, df in sheets_dict.items():
                all_text.append(f"### Sheet Name: {sheet_name}")
                all_text.append(df.to_markdown(index=False))
                all_text.append("\n")
            return "\n".join(all_text)

        elif ext == 'csv':
            return pd.read_csv(file).to_markdown(index=False)
        else:
            return file.getvalue().decode("utf-8")
    except:
        return ""

async def safe_api_call(client, model, messages, retries=3, temperature=0, **kwargs):
    """API 호출 (RateLimit 처리 및 Temperature 설정 포함)"""
    for i in range(retries):
        try:
            # temperature 인자 추가로 일관성 확보
            return await client.chat.completions.create(
                model=model, 
                messages=messages, 
                temperature=temperature, 
                **kwargs
            )
        except RateLimitError:
            await asyncio.sleep(1 + (i * 0.5))
        except Exception as e:
            # print(f"API Error: {e}") 
            await asyncio.sleep(0.5)
    return None

async def generate_output_lite(client, model, context, prompt):
    """참가자의 프롬프트를 실행하여 결과 생성"""
    messages = [
        {"role": "system", "content": "당신은 데이터 처리 엔진입니다. 지시에 따라 결과를 출력하세요."},
        {"role": "user", "content": f"---[Input File]---\n{context}\n\n---[User Prompt]---\n{prompt}"}
    ]
    # 실행 시에도 temperature=0을 주어 변동성을 최소화
    resp = await safe_api_call(client, model, messages, temperature=0)
    return resp.choices[0].message.content if resp else "Error"

# [핵심 변경] 개선된 심사 로직 (CoT 적용, 일반화 평가 추가)
async def audit_submission_lite(client, model, target, out1, original_prompt):
    judge_prompt = f"""
    당신은 'DB Inc 프롬프팅 경진대회'의 공정하고 분석적인 수석 심사위원입니다.
    제출된 프롬프트와 실행 결과를 분석하여, 아래 [판단 로직]에 따라 적절한 평가 기준표를 선택해 엄격히 채점하십시오.
    
    [판단 로직: 평가 기준표 선택]
    - 평가 기준표 1 (데이터 처리): Target 데이터가 엑셀, CSV 형식이거나, 명확한 행/열 구조를 가진 표 데이터인 경우 적용. (정답 데이터와의 일치 여부가 중요)
    - 평가 기준표 2 (일반 생성/논리): Target이 줄글, 요약, 아이디어 제안, 분류 등 비정형 텍스트인 경우 적용. (논리적 구조와 프롬프트 엔지니어링 기술이 중요)
    
    [평가 프로세스]
    1. 먼저 제출된 '참가자 Prompt'가 단순히 정답 텍스트를 그대로 출력하도록 유도하거나, 특정 데이터에만 과적합(Overfitting)된 하드코딩인지 분석하십시오.
    2. '실행 결과'가 '목표 산출물'의 핵심 의도를 달성했는지 의미론적(Semantic)으로 비교하십시오. (단순 문자열 일치 여부가 아님)
    3. '참가자 Prompt' 내부에 일반화(Generalization)를 위한 장치(예외 처리, 명확한 구분자, few-shot 예시 등)가 포함되어 있는지 분석하십시오.
    4. '참가자 Prompt' 가 상세하고 예외처리를 많이 할 수록 높은 점수를 가질 확률이 올라갑니다. 
    5. 아래 JSON 포맷으로만 결과를 출력하십시오.

    
    [데이터 포맷 및 멀티 시트 평가 지침]
    1. 포맷 유연성: Target은 엑셀 형식이지만, 참가자는 텍스트(Markdown 표, CSV, JSON 등)로 제출합니다. 데이터 값과 구조가 논리적으로 일치하면 정답으로 인정하세요.
    2. 멀티 시트(Multi-Sheet) 확인: Target 데이터에 'Sheet1', 'Sheet2' 등 여러 시트가 포함되어 있다면, 참가자의 결과물이 모든 시트의 핵심 데이터를 포함하고 있는지 확인하세요. (별도의 표로 나누거나, 하나로 잘 합쳤는지 확인)
    
    [평가 기준표 1 - 총 100점]
    
    1. 정확성 (Accuracy) - 50점
       - 50점: Target(Sheet1, Sheet2 포함)의 모든 데이터 값과 계산 결과가 정확하게 일치함.
       - 30점: 값은 대부분 맞으나, 일부 시트의 데이터가 누락되거나 오차가 있음.
       - 20점 이하: 핵심 데이터가 틀리거나, 특정 시트 내용을 통째로 누락함.

    2. 명확성 (Clarity) - 30점
       - 30점: 프롬프트가 "어떤 데이터를 어떻게 가공하여 어떤 형식으로 출력하라"는 지시가 매우 구체적임.
       - 20점: 지시가 다소 추상적이거나("알아서 정리해"), 단계가 모호함.
       - 10점 이하: 원하는 바를 파악하기 어려움.

    3. 규칙 및 검증 (Format Consistency) - 20점
       - 20점: Target의 컬럼 구조나 데이터 형식을 잘 준수하였으며, 가독성이 뛰어남.
       - 10점 이하: 표가 깨지거나, 데이터 구분(쉼표, 탭 등)이 엉망이라 활용이 불가능함.


    [평가 기준표 2 - 총 100점]

    1. 결과 정확성 (Output Fidelity) - 배점 40점
       - 40점: 목표 산출물의 핵심 정보와 뉘앙스를 모두 정확히 포함함. 형식이 약간 다르더라도 의도가 완벽히 일치하면 만점.
       - 30점: 핵심 내용은 일치하나, 불필요한 서론/결론이 있거나 형식이 목표와 다소 상이함.
       - 20점 이하: 핵심 정보가 누락되거나 환각(Hallucination)이 포함됨.
       * 감점 요인: 실행 결과가 목표값과 100% 텍스트가 일치하더라도, 프롬프트가 단순 복사-붙여넣기 수준이라면 이 항목에서 감점할 것.

    2. 프롬프트 공학 기술 및 일반화 (Prompt Engineering & Robustness) - 배점 40점
       * 이 항목은 결과물이 아닌 '참가자 Prompt' 자체를 평가함.
       - 40점: 다른 입력값이 들어와도 작동하도록 논리적 구조(Step-by-step)를 갖춤. 제약 조건(Constraints), 예외 처리, 페르소나, 구분자(Delimiters) 등을 적절히 활용함.
       - 30점: 지시는 명확하지만 예외 처리나 구조적 제약이 부족하여, 입력 데이터가 바뀔 경우 오류 가능성이 있음.
       - 10점 이하: 단순히 "이거 해줘" 수준의 1차원적 지시이거나, 특정 결과값만 나오도록 강제한 경우(Overfitting).

    3. 가독성 및 명확성 (Clarity) - 배점 20점
       - 20점: 프롬프트가 구조화되어 있어 타인이 읽었을 때 의도를 즉시 파악 가능함.
       - 10점 이하: 줄글로 나열되어 있거나 지시 사항이 서로 상충됨.

    [입력 데이터]
    - 참가자 Prompt: "{original_prompt}"
    - 목표 산출물 (Target): {target[:50000]}
    - 실행 결과 (Result): {out1[:50000]}

    [출력 형식 (JSON Only)]
    {{
        "reasoning": "채점 근거를 3문장 이내로 먼저 서술하세요. (특히 일반화 가능성 위주로)",
        "score_accuracy": int,
        "score_robustness": int,
        "score_clarity": int,
        "total_score": int,
        "feedback": "참가자를 위한 피드백 (한글, 200자 이내)"
    }}
    """
    
    resp = await safe_api_call(
        client, 
        model, 
        [{"role": "system", "content": "You are a strict AI judge. Output valid JSON only."}, 
         {"role": "user", "content": judge_prompt}],
        temperature=0, # 일관성을 위해 0으로 설정 (필수)
        response_format={"type": "json_object"}
    )
    
    if not resp:
        return {"total_score": 0, "score_accuracy": 0, "score_robustness": 0, "score_clarity": 0, "reasoning": "API Error", "feedback": "Error"}

    try:
        content = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except:
        return {"total_score": 0, "score_accuracy": 0, "score_robustness": 0, "score_clarity": 0, "reasoning": "Parsing Error", "feedback": "Error"}

async def process_participant(sem, client, row, context, target, gen_model, judge_model):
    name = row.iloc[0]
    prompt = row.iloc[1]
    
    async with sem:
        try:
            # 1. 생성
            out1 = await generate_output_lite(client, gen_model, context, prompt)
            
            # 2. 심사 (개선된 함수 사용)
            score_data = await audit_submission_lite(client, judge_model, target, out1, prompt)
            
            return {
                "이름": name, 
                "총점": score_data.get('total_score', 0),
                "정확성": score_data.get('score_accuracy', 0), 
                "일반화": score_data.get('score_robustness', 0), 
                "명확성": score_data.get('score_clarity', 0), 
                "심사 근거": score_data.get('reasoning', ""),
                "피드백": score_data.get('feedback', ""),
                "결과물": out1[:100] + "..."
            }
        except Exception as e:
            return {"이름": name, "총점": 0, "피드백": f"Err: {e}", "결과물": "Fail"}

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
        
        if eta_seconds >= 60:
            eta_str = f"{eta_seconds // 60}분 {eta_seconds % 60}초"
        else:
            eta_str = f"{eta_seconds}초"
            
        progress_bar.progress(completed / total_count)
        
        status_box.markdown(f"""
        <div class='status-box'>
        🚀 채점 진행 중... ({completed} / {total_count}명)<br>
        <span class='eta-text'>⚡ 속도: {avg_speed:.2f}초/명 | ⏳ 남은 시간: {eta_str}</span>
        </div>
        """, unsafe_allow_html=True)
        
    return pd.DataFrame(results), time.time() - start_time

# ---------------------------------------------------------
# [메인] 실행
# ---------------------------------------------------------
st.title("📏 DB Inc 프롬프팅 경진대회 채점 시스템 v2.0")

if st.button("🚀 고속 채점 시작 (개선된 로직 적용)", type="primary", use_container_width=True):
    if not uploaded_context or not uploaded_target or not uploaded_participants:
        st.warning("⚠️ 파일 3개를 모두 업로드해주세요.")
    else:
        ctx = read_file(uploaded_context)
        tgt = read_file(uploaded_target)
        df_p = pd.read_excel(uploaded_participants)
        
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            res_df, total_time = asyncio.run(run_grading_system(
                api_key, ctx, tgt, df_p, concurrency_limit, gen_model, judge_model
            ))
            
            # 순위 산정
            res_df["총점"] = pd.to_numeric(res_df["총점"], errors='coerce').fillna(0)
            res_df = res_df.sort_values(by="총점", ascending=False).reset_index(drop=True)
            res_df["순위"] = res_df.index + 1
            
            st.success(f"✅ 채점 완료! (총 소요 시간: {int(total_time)}초)")
            
            # 대시보드
            st.markdown("### 📊 결과 대시보드")
            k1, k2, k3 = st.columns(3)
            k1.metric("참가자", f"{len(res_df)}명")
            k2.metric("평균 점수", f"{round(res_df['총점'].mean(), 1)}점")
            if not res_df.empty:
                k3.metric("최고 점수", f"{res_df.iloc[0]['총점']}점", res_df.iloc[0]['이름'])
            
            st.divider()
            st.caption("상위권 점수 분포")
            if not res_df.empty:
                chart_data = res_df.head(15).set_index("이름")[["총점"]]
                st.bar_chart(chart_data, color="#2e7d32")
            
            st.divider()
            st.subheader("📋 상세 성적표")
            
            # 컬럼 순서 및 설정 재정의
            display_cols = ["순위", "이름", "총점", "정확성", "일반화", "명확성", "심사 근거", "피드백"]
            
            st.dataframe(
                res_df[display_cols],
                use_container_width=True,
                column_config={
                    "총점": st.column_config.ProgressColumn("총점", min_value=0, max_value=100, format="%d점"),
                    "정확성": st.column_config.NumberColumn("정확성(40)", format="%d"),
                    "일반화": st.column_config.NumberColumn("일반화(40)", format="%d"),
                    "명확성": st.column_config.NumberColumn("명확성(20)", format="%d"),
                    "심사 근거": st.column_config.TextColumn("심사 근거 (CoT)", width="medium"),
                    "피드백": st.column_config.TextColumn("심사평", width="medium")
                }
            )
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
                worksheet = writer.sheets['Sheet1']
                workbook = writer.book
                format_wrap = workbook.add_format({'text_wrap': True})
                # 엑셀 컬럼 너비 조정
                worksheet.set_column('A:E', 10) # 점수 컬럼들
                worksheet.set_column('F:G', 50, format_wrap) # 근거, 피드백 등
                
            st.download_button("💾 최종 결과 엑셀 다운로드", output.getvalue(), "final_result_v2.xlsx", type="primary")
            
        except Exception as e:
            st.error(f"시스템 에러 발생: {e}")
            st.error("TIP: API 키를 확인하거나, 동시 채점 인원(Concurrency)을 줄여보세요.")
