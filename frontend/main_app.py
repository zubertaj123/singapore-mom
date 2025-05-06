import streamlit as st
import os
from datetime import datetime

from backend.vectorstore import load_vectorstore
from backend.qa_chain import build_qa_chain
from backend.agent_router import route_to_agent
from backend.tools import (
    dependent_pass_checker, quota_checker, license_checker,
    appointment_api_tool, wpol_application_tool, status_api_tool
)
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# === SETUP ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

st.set_page_config(page_title="Agentic AI – MOM Assistant", layout="wide")

# === SESSION STATE ===
st.session_state.setdefault("agentic_trace", [])
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("tool_logs", [])
st.session_state.setdefault("form_mode", False)
st.session_state.setdefault("submitted", False)
st.session_state.setdefault("user_role", "Individual")

# === STYLE & HEADER ===
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: #f5f7fb; max-width: 1200px; margin: auto; padding-bottom: 100px; }
</style>
<div style="background:#36312d;padding:0.6rem 1rem;color:#51f2b8;font-weight:500">
    🧠 Agentic AI – MOM Assistant (Singapore Ministry of Manpower)
</div>
""", unsafe_allow_html=True)

# === INTRO ===
st.markdown("""
<div style="background-color: #f2f4f7; padding: 1rem; border-left: 4px solid #007a5e;">
<strong>Agentic AI assistant</strong> for answering questions based on Singapore MOM policies.
</div>
""", unsafe_allow_html=True)

# === TOOLS ===
def make_tools(log_ref):
    return {
        "Individual": [Tool(name="Dependant Pass Tool", func=lambda q: dependent_pass_checker(q, log_ref), description="Check eligibility for Dependant Pass")],
        "Employer": [Tool(name="Quota Tool", func=lambda q: quota_checker(q, log_ref), description="Check S Pass quota rules")],
        "Employment Agent": [Tool(name="License Tool", func=lambda q: license_checker(q, log_ref), description="Guide for licensing")],
    }, [
        Tool(name="Appointment Tool", func=lambda q: appointment_api_tool(q, log_ref), description="Simulate scheduling an appointment"),
        Tool(name="WPOL Submission Tool", func=lambda q: wpol_application_tool(q, log_ref), description="Simulate work permit application"),
        Tool(name="Status Checker Tool", func=lambda q: status_api_tool(q, log_ref), description="Simulate checking application status")
    ]

# === VECTORSTORE AND CHAIN ===
vectorstore = load_vectorstore()
qa_chain = build_qa_chain(vectorstore.as_retriever(search_kwargs={"k": 4}))

# === UI ===
role = st.selectbox("🔘 Select your role:", ["Individual", "Employer", "Employment Agent"], key="user_role")

if not st.session_state.chat_history:
    st.chat_message("assistant").markdown("How can I assist you with MOM policies today?")

for msg in st.session_state.chat_history:
    st.chat_message("user").markdown(msg["question"])
    st.chat_message("assistant").markdown(msg["answer"])

query = st.chat_input("Ask your MOM-related question...")

if query:
    st.session_state.form_mode = False
    st.session_state.submitted = False

    st.session_state.agentic_trace = [("🧑‍💬 User Message", query)]
    tools_dict, backend_tools = make_tools(st.session_state.tool_logs)

    if "apply" in query.lower() and "permit" in query.lower():
        st.session_state.form_mode = True
        st.session_state.agentic_trace.append(("🧠 Routed to", "Work Permit Form"))
        st.session_state.chat_history.append({"timestamp": str(datetime.now()), "question": query, "answer": "📝 Let's proceed with the Work Permit application form."})
        st.rerun()
    else:
        st.session_state.chat_history.append({"timestamp": str(datetime.now()), "question": query, "answer": ""})
        result = route_to_agent(query, role, tools_dict, backend_tools, llm)
        if not result:
            rag_result = qa_chain.invoke({"question": query, "role": role})
            result = rag_result["result"]
            st.session_state.agentic_trace.append(("📚 Used RAG", result[:150]))
        else:
            st.session_state.agentic_trace.append(("🛠️ Used Tool", result[:150]))

        st.session_state.chat_history[-1]["answer"] = result
        st.rerun()

# === FORM ===
if st.session_state.form_mode and not st.session_state.submitted:
    with st.form("work_permit_form", clear_on_submit=True):
        name = st.text_input("Full Name")
        passport = st.text_input("Passport Number")
        nationality = st.text_input("Nationality")
        role = st.text_input("Job Title in Singapore")
        employer = st.text_input("Employer Company Name")
        location = st.text_input("Worksite Location")
        duration = st.text_input("Work Permit Duration (e.g. 2 years)")
        submit = st.form_submit_button("Submit Application")

        if submit:
            st.session_state.submitted = True
            st.session_state.form_mode = False
            ref_id = f"WP-{datetime.now().strftime('%y%m%d%H%M%S')}"
            st.chat_message("assistant").markdown(f"✅ Application submitted for **{name}**. Reference ID: **{ref_id}**.")
            st.rerun()

# === SIDEBAR ===
with st.sidebar:
    st.markdown("### 🧠 Agentic Trace Log")
    for label, value in st.session_state.agentic_trace:
        st.markdown(f"**{label}:** {value}")

# === FOOTER ===
st.markdown("""
<hr>
<p style='text-align: center; color: gray;'>
⚠️ Prototype by Virtusa using public data from <a href='https://www.mom.gov.sg' target='_blank'>mom.gov.sg</a>. Not affiliated with MOM.
</p>
""", unsafe_allow_html=True)
