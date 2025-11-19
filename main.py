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

st.title("🏆 DB Inc 프롬프팅 경진대회 리더보드")
st.markdown("### ⚡ Powered by GPT-5 Nano (Fastest & Most Cost-efficient)")

# ---------------------------------------------------------
# [사이드바] 입력 및 설정
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("OpenAI API Key 입력", type="password")
    
    st.divider()
    st.header("📂 데이터 업로드")
    uploaded_context = st.file_uploader("1. 문맥 자료 (PDF/Txt/Excel)", type=['pdf', 'txt', 'xlsx'])
    uploaded_target = st.file_uploader("2. 정답지 (Txt/Excel)", type=['txt', 'xlsx'])
    uploaded_participants = st.file_uploader("3. 참가자 명단 (Excel)", type=['xlsx'])
    
    st.info("💡 참가자 엑셀 형식: A1='이름', A2='프롬프트'")

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
    
    # 모델 설정 (GPT-5 nano)
    MODEL_ID = "gpt-5-nano" 
    
    for idx, row in participants.iterrows():
        name = row.iloc[0]
        prompt = row.iloc[1]
        
        status.write(f"⚡ **{name}**님 채점 중... ({idx+1}/{total})")
        bar.progress((idx + 1) / total)
        
        try:
            # 1. 모델 실행 (GPT-5 nano 사용)
            messages = [
                {"role": "system", "content": "당신은 유능한 한국어 AI 어시스턴트입니다."},
                {"role": "user", "content": f"문맥 데이터:\n{context}\n\n요청사항:\n{prompt}"}
            ]
            
            # 실행 1 & 2 (재현성 검증)
            out1 = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.7).choices[0].message.content
            out2 = client.chat.completions.create(model=MODEL_ID, messages=messages, temperature=0.7).choices[0].message.content
            
            # 2. 심사 (Judge - 역시 GPT-5 nano 사용)
            judge_prompt = f"""
            당신은 프롬프트 엔지니어링 대회의 엄격한 심사위원입니다.
            아래 기준표(Rubric)에 따라 채점하고, 결과는 반드시 JSON 형식으로 한국어로 작성하세요.
            
            [채점 기준]
            1. 정확성 (50점 만점): 결과물이 정답(Target)의 핵심 내용과 형식을 얼마나 정확히 맞췄는가?
            2. 명확성 (30점 만점): 프롬프트가 페르소나, 단계별 지시 등을 명확히 포함하는가?
            3. 일관성 (20점 만점): 두 번 실행(Output 1 vs 2)했을 때 결과가 얼마나 유사한가?

            [데이터]
            - 참가자 프롬프트: {prompt}
            - 정답지(Target): {target}
            - 실행결과 1: {out1}
            - 실행결과 2: {out2}
            
            [출력 포맷 (JSON Only)]
            {{
                "accuracy": 점수(int),
                "clarity": 점수(int),
                "consistency": 점수(int),
                "reasoning": "상세한 심사평을 한국어로 작성 (100자 이상)"
            }}
            """
            
            judge = client.chat.completions.create(
                model=MODEL_ID, 
                messages=[{"role": "system", "content": "JSON output only."}, {"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"}
            )
            score_data = json.loads(judge.choices[0].message.content)
            
            total_score = score_data['accuracy'] + score_data['clarity'] + score_data['consistency']
            
            results.append({
                "이름": name,
                "총점": total_score,
                "정확성": score_data['accuracy'],
                "명확성": score_data['clarity'],
                "일관성": score_data['consistency'],
                "심사평": score_data['reasoning'],
                "프롬프트": prompt,
                "결과물": out1
            })
            
        except Exception as e:
            results.append({"이름": name, "총점": 0, "심사평": f"에러 발생: {e}", "프롬프트": prompt, "결과물": "실패"})
            
    status.success("🎉 채점 완료!")
    bar.empty()
    return pd.DataFrame(results)

# ---------------------------------------------------------
# [메인] 실행 및 결과 화면
# ---------------------------------------------------------
if st.button("🚀 채점 시작하기 (GPT-5 nano)", type="primary", use_container_width=True):
    if not api_key or not uploaded_context or not uploaded_target or not uploaded_participants:
        st.error("⚠️ API 키와 모든 파일(문맥, 정답, 참가자)을 업로드해주세요.")
    else:
        with st.spinner("GPT-5 nano가 초고속으로 채점 중입니다..."):
            try:
                client = OpenAI(api_key=api_key)
                ctx_text = read_file(uploaded_context)
                tgt_text = read_file(uploaded_target)
                df_part = pd.read_excel(uploaded_participants)
                
                # 채점 실행
                raw_df = evaluate(client, ctx_text, tgt_text, df_part)
                
                # 순위 산정
                result_df = raw_df.sort_values("총점", ascending=False).reset_index(drop=True)
                result_df.index = result_df.index + 1  # 1위부터 시작
                result_df.index.name = "순위"
                
                # 1. 명예의 전당 (Top 3)
                st.divider()
                st.subheader("🥇 명예의 전당")
                col1, col2, col3 = st.columns(3)
                
                top3 = result_df.head(3)
                if len(top3) > 0:
                    col1.metric(label="🥇 1위", value=f"{top3.iloc[0]['이름']}", delta=f"{top3.iloc[0]['총점']}점")
                if len(top3) > 1:
                    col2.metric(label="🥈 2위", value=f"{top3.iloc[1]['이름']}", delta=f"{top3.iloc[1]['총점']}점")
                if len(top3) > 2:
                    col3.metric(label="🥉 3위", value=f"{top3.iloc[2]['이름']}", delta=f"{top3.iloc[2]['총점']}점")

                # 2. 전체 리더보드 (테이블)
                st.divider()
                st.subheader("📊 전체 리더보드")
                display_cols = ["이름", "총점", "정확성", "명확성", "일관성", "심사평"]
                st.dataframe(result_df[display_cols], use_container_width=True)

                # 3. 상세 분석 (Expandable)
                st.divider()
                st.subheader("🧐 참가자별 상세 결과 분석")
                for idx, row in result_df.iterrows():
                    with st.expander(f"{idx}위 - {row['이름']} (총점: {row['총점']}점)"):
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.markdown("**📝 제출한 프롬프트**")
                            st.info(row['프롬프트'])
                        with c2:
                            st.markdown("**🤖 AI 실행 결과**")
                            st.success(row['결과물'])
                        
                        st.markdown("---")
                        st.markdown(f"**👮 심사위원 평가:** {row['심사평']}")
                        st.caption(f"세부 점수: 정확성 {row['정확성']} + 명확성 {row['명확성']} + 일관성 {row['일관성']}")

                # 4. 엑셀 다운로드
                st.divider()
                out_io = io.BytesIO()
                with pd.ExcelWriter(out_io, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, sheet_name="채점결과")
                st.download_button("💾 전체 결과 엑셀 다운로드", out_io.getvalue(), "GPT5_채점결과.xlsx", type="primary")
            
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                st.warning("혹시 오류가 계속되면, OpenAI API 키에 'gpt-5-nano' 사용 권한이 있는지 확인해주세요.")
