import streamlit as st
import pandas as pd
import json
import os
import asyncio
from openai import AsyncOpenAI, RateLimitError
from pypdf import PdfReader
import io

# ---------------------------------------------------------
# [설정] 페이지 기본 세팅
# ---------------------------------------------------------
st.set_page_config(page_title="DB Inc 프롬프팅 대회 채점기", layout="wide", page_icon="🎓")

# Railway 환경변수 로드
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# [스타일]
# ---------------------------------------------------------
st.markdown("""
    <style>
    .metric-container { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .guide-box { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [사이드바]
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 시스템 설정")
    
    if api_key:
        st.success(f"✅ API Key 연동 완료\n(GPT-5 nano / Async)")
    else:
        st.error("❌ API Key가 없습니다. Railway 설정을 확인하세요.")
        st.stop()
    
    st.divider()
    
    # 속도 조절
    concurrency_limit = st.slider(
        "동시 채점 인원 (명)", 
        min_value=1, max_value=10, value=5,
        help="안전한 채점을 위해 5명을 권장합니다."
    )
    
    st.divider()
    
    st.subheader("📂 데이터 업로드")
    uploaded_context = st.file_uploader("1. 문맥 자료 (PDF/Txt/Excel)", type=['pdf', 'txt', 'xlsx'])
    uploaded_target = st.file_uploader("2. 정답지 (Txt/Excel)", type=['txt', 'xlsx'])
    uploaded_participants = st.file_uploader("3. 참가자 명단 (Excel)", type=['xlsx'])
    
    st.divider()
    
    if st.button("🧪 테스트용 샘플(10명) 다운로드"):
        dummy_data = {
            "이름": [
                "1.김고수(완벽)", "2.이대충(부족)", "3.박평범(무난)", "4.최구체(상세)", "5.정질문(모호)",
                "6.강포맷(형식)", "7.조단답(짧음)", "8.윤논리(CoT)", "9.장영어(영문)", "10.임창의(독특)"
            ],
            "프롬프트": [
                "너는 15년 차 수석 데이터 분석가야. 경영진 보고를 위해 첨부된 파일의 [매출 추이]와 [감소 원인]을 분석해줘. 출력은 반드시 Markdown 표 형식으로 작성하고, 마지막에 '전략적 제언' 3가지를 글머리 기호로 추가해.",
                "이거 요약 좀.",
                "파일 내용을 읽고 중요한 내용을 3줄로 요약해주세요. 말투는 공손하게 해주세요.",
                "데이터를 분석해서 JSON 형식으로 출력해줘. Key값은 'issue', 'cause', 'solution'으로 구성하고, 내용은 한국어로 채워줘.",
                "이 파일에서 가장 중요한 게 뭐야? 그리고 왜 중요한지 설명해줄 수 있어?",
                "첨부 자료를 바탕으로 주간 업무 보고서를 작성해. [개요] - [상세 실적] - [특이 사항] 순서로 목차를 잡고 작성해줘.",
                "내용 다 필요 없고, 결론만 한 문장으로 말해.",
                "먼저 데이터를 전체적으로 훑어보고 이상치를 찾아내. 그 다음 이상치가 발생한 이유를 추론해보고, 최종적으로 해결책을 제시해. 생각의 과정을 단계별로(Step-by-step) 보여줘.",
                "Analyze the provided file and summarize the key findings in English. Use professional business terminology.",
                "너는 비판적인 투자자야. 이 자료를 보고 투자를 할지 말지 결정하려고 해. 자료의 논리적 허점이나 부족한 데이터를 날카롭게 지적해줘."
            ]
        }
        df_dummy = pd.DataFrame(dummy_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_dummy.to_excel(writer, index=False)
        st.download_button("📥 샘플 엑셀(10명) 받기", output.getvalue(), "participants_sample_10.xlsx")

# ---------------------------------------------------------
# [함수] 파일 읽기 및 API 호출
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
            wait_time = (i + 1) * 2
            await asyncio.sleep(wait_time)
        except Exception as e:
            raise e
    return None

async def evaluate_single_participant(sem, client, idx, row, context, target, status_text, total):
    name = row.iloc[0]
    prompt = row.iloc[1]
    MODEL_NAME = "gpt-5-nano"
    
    async with sem:
        try:
            status_text.write(f"🔄 **{name}**님 채점 진행 중... ({idx+1}/{total})")
            
            # 1. 실행 (Generation)
            messages = [
                {"role": "system", "content": "데이터 분석 AI입니다. 주어진 Context를 바탕으로 사용자의 Prompt를 수행하세요."},
                {"role": "user", "content": f"---[Context]---\n{context}\n\n---[Prompt]---\n{prompt}"}
            ]
            
            task1 = safe_api_call(client, MODEL_NAME, messages)
            task2 = safe_api_call(client, MODEL_NAME, messages)
            resp1, resp2 = await asyncio.gather(task1, task2)
            
            if not resp1 or not resp2: raise Exception("API 호출 실패")
            out1 = resp1.choices[0].message.content
            out2 = resp2.choices[0].message.content
            
            # 2. 심사 (Judge) - ★여기가 핵심입니다!★
            # 심사 기준을 원문 그대로 상세하게 주입합니다.
            judge_prompt = f"""
            당신은 프롬프트 경진대회 심사위원입니다. 
            아래 [상세 평가 기준]을 **빠짐없이 꼼꼼하게 대조하여** 점수를 매기세요.
            단, JSON 결과의 **'reasoning' 필드는 엑셀 저장을 위해 200자 이내로 핵심만 요약**해서 작성하세요.
            
            [상세 평가 기준]
            1. 정확성 (Accuracy) - 배점 50점
               - 50점: 프롬프트 실행 결과가 목표 산출물(Target)과 내용/형식 모두 일치하며, 불필요한 오류/누락 없이 완전하게 재현됨.
               - 30점: 핵심 내용은 동일하나 세부 표현/구조에서 일부 차이 또는 부분적 누락이 있음.
               - 20점 이하: 주요 내용이 누락되거나 결과 구조가 달라 목표 산출물과 불일치.
               
            2. 명확성 (Prompt Clarity) - 배점 30점
               - 30점: 프롬프트가 명확한 역할 지시(예: "너는 데이터 분석가이다")와 단계별 요구사항을 포함하고, 사람이 읽어도 논리적/직관적으로 이해 가능함.
               - 20점: 지시문은 이해 가능하나 일부 모호한 표현 또는 불명확한 조건으로 인해 출력 변동 가능성이 있음.
               - 10점 이하: 구조가 불분명하거나 지시 문장이 혼합되어 AI가 의도를 일관되게 해석하기 어려움.
               
            3. 규칙 및 검증 (Consistency) - 배점 20점
               - 20점: 동일 조건에서 재실행(Out1 vs Out2) 시 동일한 결과를 도출하며, 테스트/비교 등을 통해 안정성을 입증함.
               - 15점: 재실행 시 경미한 변동이 있으나 전반적 구조와 내용은 유지됨.
               - 10점 이하: 일관성 확인 절차가 부족하거나, 실행마다 결과가 상이하여 재현성 낮음.

            [평가할 데이터]
            - 참가자 프롬프트: {prompt}
            - 목표 산출물(Target): {target}
            - 실행 결과 1 (Out1): {out1}
            - 실행 결과 2 (Out2): {out2}
            
            [출력 포맷 (JSON Only)]
            {{
                "accuracy": 점수(int),
                "clarity": 점수(int),
                "consistency": 점수(int),
                "reasoning": "위 평가 기준에 근거한 구체적인 심사평 (반드시 200자 이내 요약, 한국어)"
            }}
            """
            
            judge_resp = await safe_api_call(client, MODEL_NAME, 
                [{"role": "system", "content": "JSON output only."}, {"role": "user", "content": judge_prompt}])
            
            score_data = json.loads(judge_resp.choices[0].message.content)
            total_score = score_data['accuracy'] + score_data['clarity'] + score_data['consistency']
            
            return {
                "이름": name, "총점": total_score,
                "정확성": score_data['accuracy'], "명확성": score_data['clarity'], "재현성": score_data['consistency'],
                "심사평": score_data['reasoning'], "실행결과": out1
            }
        except Exception as e:
            return { "이름": name, "총점": 0, "정확성": 0, "명확성": 0, "재현성": 0, "심사평": f"Error: {str(e)}", "실행결과": "Fail" }

async def run_all_evaluations(api_key, context, target, df_participants, limit):
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(limit) 
    tasks = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(df_participants)
    
    for idx, row in df_participants.iterrows():
        task = evaluate_single_participant(sem, client, idx, row, context, target, status_text, total)
        tasks.append(task)
    
    status_text.info(f"🚀 채점 시작! (동시 처리: {limit}명)")
    results = []
    completed = 0
    
    for f in asyncio.as_completed(tasks):
        result = await f
        results.append(result)
        completed += 1
        progress_bar.progress(completed / total)
        
    status_text.success("✅ 모든 채점 완료!")
    return pd.DataFrame(results)

# ---------------------------------------------------------
# [메인] UI
# ---------------------------------------------------------
st.title("🏆 DB Inc 프롬프팅 경진대회 채점 시스템")

# 사용 가이드
with st.expander("📘 사용 가이드 및 채점 기준 확인하기", expanded=True):
    st.markdown("#### 🚀 사용 순서")
    st.markdown("1. 왼쪽 사이드바에서 **[🧪 테스트용 샘플(10명) 다운로드]** 버튼을 눌러 엑셀을 받으세요.")
    st.markdown("2. 받은 엑셀을 **[3. 참가자 명단]**에 업로드하세요. (문맥/정답 파일도 업로드 필요)")
    st.markdown("3. **[채점 시작]** 버튼을 누르면 10명을 동시에 채점합니다.")

st.divider()

# 실행 로직
if st.button("🚀 채점 시작하기 (Start Grading)", type="primary", use_container_width=True):
    if not uploaded_context or not uploaded_target or not uploaded_participants:
        st.warning("⚠️ 왼쪽 사이드바에서 모든 파일(3개)을 먼저 업로드해주세요!")
    else:
        ctx = read_file(uploaded_context)
        tgt = read_file(uploaded_target)
        df_p = pd.read_excel(uploaded_participants)
        
        try:
            res_df = asyncio.run(run_all_evaluations(api_key, ctx, tgt, df_p, concurrency_limit))
            
            res_df = res_df.sort_values(by="총점", ascending=False).reset_index(drop=True)
            res_df["순위"] = res_df.index + 1
            
            # Dashboard
            st.markdown("### 📊 채점 결과 대시보드")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("👥 참가자", f"{len(res_df)}명")
            k2.metric("📈 평균 점수", f"{round(res_df['총점'].mean(), 1)}점")
            k3.metric("🥇 1위", f"{res_df.iloc[0]['이름']}", f"{res_df.iloc[0]['총점']}점")
            
            st.divider()
            st.caption("상위 10명 점수 그래프")
            st.bar_chart(res_df.head(10).set_index("이름")["총점"], color="#00CC96")
            
            st.divider()
            st.subheader("📋 전체 리더보드")
            st.dataframe(
                res_df[["순위","이름","총점","정확성","명확성","재현성","심사평"]], 
                use_container_width=True,
                column_config={
                    "총점": st.column_config.ProgressColumn("총점", min_value=0, max_value=100, format="%d점"),
                    "심사평": st.column_config.TextColumn("심사평 (200자 요약)")
                }
            )
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                res_df.to_excel(writer, index=False)
                worksheet = writer.sheets['Sheet1']
                worksheet.set_column('G:G', 70) # 200자니까 너비 더 넓게
                
            st.download_button(
                label="💾 결과 엑셀 다운로드", 
                data=output.getvalue(), 
                file_name="grading_result_10users.xlsx",
                type="primary"
            )
            
        except Exception as e:
            st.error(f"시스템 에러가 발생했습니다: {e}")
