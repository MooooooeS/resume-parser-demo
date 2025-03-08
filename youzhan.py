import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_match_score(resume_text, job_description):
    embeddings_resume = model.encode(resume_text, convert_to_tensor=True)
    embeddings_job = model.encode(job_description, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings_resume, embeddings_job).item()
    return round(similarity_score * 100, 2)

st.title("智能简历解析与匹配")
resume_text = st.text_area("输入简历内容：")
job_description = st.text_area("输入职位描述：")

if st.button("计算匹配度"):
    score = calculate_match_score(resume_text, job_description)
    st.write(f"匹配度评分: {score}%")
