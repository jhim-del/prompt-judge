import streamlit as st
import pandas as pd
import json
import os
from openai import OpenAI
from pypdf import PdfReader
import io

# ---------------------------------------------------------
# [설정] 페이지 및 API 키 자동 로드
# ---------------------------------------------------------
st.set_page_config(page_title="DB Inc 프롬프팅 대회 채점기", layout="wide", page_icon="🏆")

# Railway 환경변수에서 API 키를 가져옵니다.
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# [사이드바] 파일 업로드
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 시스템 상태")
    if api_key:
        st.success("✅ API Key가 Railway에서 로드되었습니다.")
    else:
        st.error("❌ API Key를 찾을 수 없습니다. Railway Variables를 확인해주세요.")
        st.stop() # 키 없으면 실행 중단
    
    st.divider()
    st.header("📂 데이터 업로드")
    uploaded_context = st.file_uploader("1. 문맥 자료 (PDF/Txt/Excel)", type=['pdf', 'txt', 'xlsx'])
    uploaded_target = st.file_uploader("2. 정답지 (Txt/Excel)", type=['txt', 'xlsx'])
    uploaded_participants = st.file_uploader("3. 참가자 명단 (Excel)", type=['xlsx'])
    
    st.info("💡 참가자 엑셀 형식: A1='이름', A2='프롬프트'")

# ---------------------------------------------------------
# [함수] 파일 처리 및 채점 로직
# ---------------------------------------------------------
def read_file(file):
    if not file: return None
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        reader = PdfReader(file)
        return "".join([page.extract_text() for page in reader.pages])
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(file).to_markdown(index=False)
    else:
        return file.getvalue().decode("utf-8")

def evaluate(client, context, target, participants):
    results = []
    bar = st.progress(0)
    status = st.empty()
    total = len(participants)
    
    # 사용할 모델 설정 (혹시 5-nano가 안 되면 gpt-4o로 자동 변경 권장)
    # 현재 코드는 사용자 요청대로 설정됨
    MODEL_NAME = "gpt-5-nano" 
    
    for idx, row in participants.iterrows():
        name = row.iloc[0]
        prompt = row.iloc[1]
        
        status.write(f"⚡ **{name}**님 평가 진행 중... ({idx+1}/{total})")
        bar.progress((idx + 1) / total)
        
        try:
            # ====================================================
            # 1단계: 참가자의 프롬프트 실행 (Generation)
            # ====================================================
            # 문맥 파일 + 참가자 프롬프트를 합쳐서 GPT에 입력
            messages = [
                {"role": "system", "content": "당신은 데이터 분석 어시스턴트입니다. 제공된 Context를 바탕으로 사용자의 요청을 수행하세요."},
                {"role": "user", "content": f"---[Context File]---\n{context}\n\n---[User Prompt]---\n{prompt}"}
            ]
            
            # 재현성(Consistency) 검증을 위해 2번 실행
            out1 = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.7).choices[0].message.content
            out2 = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=0.7).choices[0].message.content
            
            # ====================================================
            # 2단계: 심사 및 채점 (Evaluation)
            # ====================================================
            # 보내주신 채점표(이미지)의 기준을 정확히 반영
            judge_prompt = f"""
            당신은 프롬프트 경진대회의 심사위원입니다. 
            아래의 [평가 기준]에 맞춰 참가자를 채점하고 JSON 형식으로 응답하세요.
            
            [평가 기준표]
            1. 정확성 (Accuracy) - 배점 50점
               - 50점: 결과가 목표 산출물(Target)과 내용/형식 모두 일치. 오류/누락 없음.
               - 30점: 핵심 내용은 동일하나 세부 표현/구조에 차이 또는 부분 누락 있음.
               - 20점 이하: 주요 내용 누락 또는 결과 구조가 목표와 불일치.
               
            2. 명확성 (Prompt Clarity) - 배점 30점
               - 30점: 명확한 역할 지시(페르소나)와 단계별 요구사항 포함. 논리적/직관적임.
               - 20점: 이해 가능하나 모호한 표현 존재, 출력 변동 가능성 있음.
               - 10점 이하: 구조 불분명, 지시 혼합으로 의도 파악 어려움.
               
            3. 규칙 및 검증 (Consistency) - 배점 20점
               - 20점: 2회 실행 결과(Output 1, 2)가 동일/유사하여 안정성 입증.
               - 15점: 경미한 변동이 있으나 전반적 구조 유지.
               - 10점 이하: 실행마다 결과가 상이하여 재현성 낮음.

            [평가 데이터]
            - 참가자 프롬프트: {prompt}
            - 목표 산출물(Target): {target}
            - 실제 결과 1: {out1}
            - 실제 결과 2: {out2}
            
            [출력 형식 (JSON)]
            {{
                "accuracy": 점수(int),
                "clarity": 점수(int),
                "consistency": 점수(int),
                "reasoning": "심사평(한글로 작성)"
            }}
            """
            
            judge = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[{"role": "system", "content": "JSON output only."}, {"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"}
            )
            score_data = json.loads(judge.choices[0].message.content)
            
            total_score = score_data['accuracy'] + score_data['clarity'] + score_data['consistency']
            
            results.append({
                "순위": 0, # 나중에 계산
                "이름": name,
                "총점": total_score,
                "정확성(50)": score_data['accuracy'],
                "명확성(30)": score_data['clarity'],
                "규칙성(20)": score_data['consistency'],
                "심사평": score_data['reasoning'],
                "실행결과": out1 # 결과 미리보기
            })
            
        except Exception as e:
            # 에러 발생 시 0점 처리
            results.append({
                "순위": 0, "이름": name, "총점": 0, 
                "정확성(50)": 0, "명확성(30)": 0, "규칙성(20)": 0,
                "심사평": f"채점 중 에러 발생: {str(e)}", "실행결과": "Error"
            })
            
    status.success("🎉 모든 채점이 완료되었습니다!")
    bar.empty()
    return pd.DataFrame(results)

# ---------------------------------------------------------
# [메인] UI 구성
# ---------------------------------------------------------
st.title("🏆 DB Inc 프롬프팅 경진대회 채점 시스템")
st.markdown("### 🤖 AI(GPT-5 nano) 기반 자동 심사 리더보드")

if st.button("🚀 채점 시작 (Start Grading)", type="primary", use_container_width=True):
    if not uploaded_context or not uploaded_target or not uploaded_participants:
        st.error("⚠️ 모든 파일(문맥, 정답, 참가자)을 업로드해주세요!")
    else:
        with st.spinner("심사위원들이 채점을 진행 중입니다... 잠시만 기다려주세요."):
            client = OpenAI(api_key=api_key)
            
            # 파일 읽기
            ctx_txt = read_file(uploaded_context)
            tgt_txt = read_file(uploaded_target)
            df_part = pd.read_excel(uploaded_participants)
            
            # 평가 실행
            result_df = evaluate(client, ctx_txt, tgt_txt, df_part)
            
            # 순위 매기기
            result_df = result_df.sort_values(by="총점", ascending=False).reset_index(drop=True)
            result_df["순위"] = result_df.index + 1
            
            # 컬럼 순서 정리
            cols = ["순위", "이름", "총점", "정확성(50)", "명확성(30)", "규칙성(20)", "심사평", "실행결과"]
            result_df = result_df[cols]

            # 1. 상위권 발표
            st.divider()
            st.subheader("🥇 명예의 전당")
            top3 = result_df.head(3)
            c1, c2, c3 = st.columns(3)
            if len(top3) > 0: c1.metric("🥇 1위", top3.iloc[0]['이름'], f"{top3.iloc[0]['총점']}점")
            if len(top3) > 1: c2.metric("🥈 2위", top3.iloc[1]['이름'], f"{top3.iloc[1]['총점']}점")
            if len(top3) > 2: c3.metric("🥉 3위", top3.iloc[2]['이름'], f"{top3.iloc[2]['총점']}점")
            
            # 2. 전체 리스트
            st.divider()
            st.subheader("📊 전체 채점 결과")
            st.dataframe(result_df, use_container_width=True)
            
            # 3. 엑셀 다운로드
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 결과 엑셀 다운로드",
                data=output.getvalue(),
                file_name="최종채점결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
