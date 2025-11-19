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
st.set_page_config(page_title="DB Inc 프롬프팅 대회 채점기", layout="wide", page_icon="🏆")

st.title("🏆 DB Inc 프롬프팅 경진대회 자동 채점 시스템")
st.markdown("### 🤖 AI 심판관이 공정하게 채점합니다")

# ---------------------------------------------------------
# [사이드바] 입력 및 설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("OpenAI API Key 입력", type="password", help="sk-... 로 시작하는 키를 입력하세요")
    
    st.divider()
    st.header("📂 파일 업로드")
    uploaded_context = st.file_uploader("1. 문맥 자료 (PDF/Txt/Excel)", type=['pdf', 'txt', 'xlsx'])
    uploaded_target = st.file_uploader("2. 정답지 (Txt/Excel)", type=['txt', 'xlsx'])
    uploaded_participants = st.file_uploader("3. 참가자 명단 (Excel)", type=['xlsx'])
    
    st.info("💡 참가자 명단 엑셀 형식: A1='이름', A2='프롬프트'")

    st.divider()
    st.subheader("🧪 테스트 데이터 생성")
    if st.button("테스트용 엑셀 파일 만들기"):
        # 테스트용 데이터 생성 로직
        data = {
            "이름": ["홍길동", "이순신", "강감찬"],
            "프롬프트": [
                "너는 데이터 분석가야. 첨부된 파일 내용을 요약해줘.", 
                "전문가로서 핵심만 3줄로 요약해.", 
                "그냥 대충 요약해줘."
            ]
        }
        df_test = pd.read_json(json.dumps(data)) # JSON 변환 후 DF 생성 (호환성)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame(data).to_excel(writer, index=False)
        
        st.download_button(
            label="📥 테스트 참가자 파일 다운로드",
            data=output.getvalue(),
            file_name="participants_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------------------------------------------------------
# [기능] 파일 처리 및 채점 로직
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
    
    for idx, row in participants.iterrows():
        name = row.iloc[0]
        prompt = row.iloc[1]
        
        status.write(f"⏳ **{name}**님 채점 중... ({idx+1}/{total})")
        bar.progress((idx + 1) / total)
        
        try:
            # 1. 실행 (재현성 확인을 위해 2회 반복)
            messages = [
                {"role": "system", "content": "You are a helpful assistant analyzing data."},
                {"role": "user", "content": f"Context:\n{context}\n\nRequest:\n{prompt}"}
            ]
            out1 = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.7).choices[0].message.content
            out2 = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.7).choices[0].message.content
            
            # 2. 채점 (Judge)
            judge_prompt = f"""
            You are a strict judge for a prompt engineering competition.
            Evaluate based on the Rubric below and return ONLY JSON.
            
            [Rubric]
            1. Accuracy (50pts): Result matches Target Output in content/format. (50=Perfect, 30=Minor diff, <20=Fail)
            2. Clarity (30pts): Persona specified? Step-by-step instructions? (30=Clear, 20=Vague, <10=Confusing)
            3. Consistency (20pts): Output 1 vs Output 2 similarity. (20=Identical, 15=Similar, <10=Different)

            [Data]
            - Prompt: {prompt}
            - Target: {target}
            - Output 1: {out1}
            - Output 2: {out2}
            
            Return JSON: {{"accuracy": int, "clarity": int, "consistency": int, "comment": "string"}}
            """
            
            judge = client.chat.completions.create(
                model="gpt-4o", 
                messages=[{"role": "system", "content": "JSON output only."}, {"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"}
            )
            score_data = json.loads(judge.choices[0].message.content)
            
            total_score = score_data['accuracy'] + score_data['clarity'] + score_data['consistency']
            
            results.append({
                "이름": name,
                "총점": total_score,
                "정확성(50)": score_data['accuracy'],
                "명확성(30)": score_data['clarity'],
                "규칙성(20)": score_data['consistency'],
                "심사평": score_data['comment'],
                "결과물": out1[:200]+"..."
            })
            
        except Exception as e:
            results.append({"이름": name, "총점": 0, "심사평": f"Error: {e}"})
            
    status.success("✅ 채점 완료!")
    bar.empty()
    return pd.DataFrame(results)

# ---------------------------------------------------------
# [메인] 실행 버튼 및 결과 표시
# ---------------------------------------------------------
if st.button("🚀 채점 시작하기", type="primary", use_container_width=True):
    if not api_key or not uploaded_context or not uploaded_target or not uploaded_participants:
        st.error("⚠️ API 키와 모든 파일을 업로드해주세요.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            ctx_text = read_file(uploaded_context)
            tgt_text = read_file(uploaded_target)
            df_part = pd.read_excel(uploaded_participants)
            
            result_df = evaluate(client, ctx_text, tgt_text, df_part)
            
            st.subheader("🥇 명예의 전당")
            st.dataframe(result_df.sort_values("총점", ascending=False).head(3), hide_index=True)
            
            st.subheader("📊 전체 결과")
            st.dataframe(result_df, hide_index=True)
            
            # 엑셀 다운로드
            out_io = io.BytesIO()
            with pd.ExcelWriter(out_io, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False)
            st.download_button("💾 채점 결과 엑셀 저장", out_io.getvalue(), "result.xlsx")
            
        except Exception as e:
            st.error(f"오류 발생: {e}")
