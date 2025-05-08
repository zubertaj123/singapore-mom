import streamlit as st
import requests
from datetime import datetime

BACKEND_URL = "http://0.0.0.0:8000"  # Adjust if your FastAPI app runs elsewhere

st.set_page_config(page_title="Agentic AI – MOM Assistant", layout="wide")

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("agentic_trace", [])
st.session_state.setdefault("user_role", "Individual")
st.session_state.setdefault("pending_response", False)
st.session_state.setdefault("form_mode", False)
st.session_state.setdefault("submitted", False)
st.session_state.setdefault("openai_api_key", "")

# === TOP BAR STYLING === 
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.st-emotion-cache-18ni7ap {visibility: hidden;}

.stApp {
    background: #f5f7fb;
    max-width: 1200px;
    margin: auto;
    padding-bottom: 100px;
}

/* Ensure block container leaves room for fixed top bar */
.block-container {
    padding-top: 5rem;
    padding-left: 16rem !important; /* leave room for sidebar */
}

/* Sticky Sidebar Alignment */
section[data-testid="stSidebar"] {
    position: fixed !important;
    top: 4.5rem !important;
    left: 0;
    width: 15rem;
    height: 100%;
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
    padding: 1rem;
    z-index: 99;
    overflow-y: auto;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #007a5e;
}

section[data-testid="stSidebar"] button {
    background-color: #007a5e !important;
    color: white !important;
    border-radius: 6px;
    margin: 4px 0;
    font-size: 0.88rem;
}

/* Chat bubbles */
.user-message {
    background: #007a5e;
    color: white;
    border-radius: 15px 15px 0 15px;
    padding: 1rem; margin: 0.5rem 0;
    max-width: 80%; margin-left: auto;
}
.bot-message {
    background: #ffffff;
    color: #202124;
    border-radius: 15px 15px 15px 0;
    padding: 1rem; margin: 0.5rem 0;
    max-width: 80%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Footer */
.footer {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #f5f7fb;
    padding: 0.3rem;
    text-align: center;
    border-top: 1px solid #e0e0e0;
    z-index: 100;
}

/* Top Bar */
.top-bar {
    position: fixed;
    top: 0; left: 0; right: 0;
    background: #36312d;
    padding: 0.6rem 1rem;
    display: flex; align-items: center;
    border-bottom: 1px solid #e0e0e0;
    z-index: 102;
}
.top-bar img {
    height: 40px;
    margin-right: 10px;
}
.top-bar-text {
    font-size: 0.95rem;
    color: #51f2b8;
    font-weight: 500;
}
            
</style>

<div class="top-bar">
    <img src="https://www.trueblueadvisory.com/wp-content/uploads/2022/06/virtusa-big-logo.png">
    <span class="top-bar-text">Agentic AI Prototype – Singapore Ministry of Manpower</span>
</div>

<style>
.spinner-container {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 1.5rem 0;
}

.pulse-ring {
    display: inline-block;
    width: 30px;
    height: 30px;
    margin-right: 10px;
    border-radius: 50%;
    background: #007a5e;
    animation: pulse 1s infinite ease-in-out;
}

@keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.5); opacity: 0.5; }
    100% { transform: scale(1); opacity: 1; }
}

.spinner-text {
    font-size: 1rem;
    color: #007a5e;
    font-weight: 500;
}
</style>
     

""", unsafe_allow_html=True)

# === INTRO TEXT ===
st.markdown("""
<div style="background-color: #f2f4f7; padding: 1rem 1.5rem; border-left: 4px solid #007a5e; margin-bottom: 1.5rem; font-size: 0.95rem; line-height: 1.6; color: #333; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
<strong style="color:#007a5e;">An Agentic AI–powered assistant</strong> that interacts with public data from the <strong>Singapore Ministry of Manpower</strong> to demonstrate Virtusa’s capabilities in autonomous decision-making and public sector automation.
</div>
""", unsafe_allow_html=True)

# === INTENT CLASSIFIER ===
def classify_form_intent(query):
    q = query.lower()
    trace = [("🧑‍💬 User Message", query)]

    if all(kw in q for kw in ["apply", "permit", "work"]) or all(kw in q for kw in ["apply", "work", "pass"]):
        trace += [
            ("🧠 Orchestrator Agent", "Intent classified as transactional – Work Pass application"),
            ("🛠️ Tool Agent", "Triggered: Work Permit Application Tool (WP Online)"),
            ("✅ Final Response", f"Work Permit form shown at {datetime.now().strftime('%H:%M:%S')}")
        ]
        return "work_pass", trace

    elif all(kw in q for kw in ["apply", "permit", "employment"]) or all(kw in q for kw in ["apply", "employment", "pass"]):
        trace += [
            ("🧠 Orchestrator Agent", "Intent classified as transactional – Employment Pass application"),
            ("🛠️ Tool Agent", "Triggered: Employment Pass Application Tool (EP eService)"),
            ("✅ Final Response", f"Employment Pass form shown at {datetime.now().strftime('%H:%M:%S')}")
        ]
        return "employment_pass", trace

    return None, trace

# === CHAT INTERFACE ===
if not st.session_state.chat_history:
    st.chat_message("assistant").markdown("<div class='bot-message'>How can I help you today?<br><span style='color:#5f6368;'>Ask about work passes, salaries, employment laws, or worker rights.</span></div>", unsafe_allow_html=True)

for entry in st.session_state.chat_history:
    st.chat_message("user").markdown(f"<div class='user-message'>{entry['question']}</div>", unsafe_allow_html=True)
    if entry["answer"]:
        st.chat_message("assistant").markdown(f"<div class='bot-message'>{entry['answer']}</div>", unsafe_allow_html=True)

query = st.chat_input("Ask a question about MOM regulations...")

if query:
    form_type, trace = classify_form_intent(query)
    st.session_state.agentic_trace = trace
    st.session_state.submitted = False

    if form_type == "work_pass":
        st.session_state.form_mode = "work"
        st.session_state.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": "📝 Let's proceed with the Work Permit application form."
        })
    elif form_type == "employment_pass":
        st.session_state.form_mode = "emp"
        st.session_state.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": "📝 Let's proceed with the Employment Pass application form."
        })
    else:
        st.session_state.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": ""
        })
        st.session_state.pending_response = True

    st.rerun()

# === FORM HANDLING ===
if st.session_state.form_mode and not st.session_state.submitted:
    with st.form("application_form", clear_on_submit=True):
        name = st.text_input("Full Name")
        passport = st.text_input("Passport Number")
        nationality = st.text_input("Nationality")
        job_title = st.text_input("Job Title in Singapore")
        employer = st.text_input("Employer Company Name")
        location = st.text_input("Worksite Location")
        duration = st.text_input("Permit Duration (e.g. 2 years)")
        submit = st.form_submit_button("Submit Application")

        if submit:
            form_label = "WP" if st.session_state.form_mode == "work" else "EP"
            ref_id = f"{form_label}-{datetime.now().strftime('%y%m%d%H%M%S')}"
            st.session_state.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": f"{form_label} Form Submission",
                "answer": f"✅ Application submitted for **{name}**. Reference ID: **{ref_id}**. You will be notified once processed."
            })
            st.session_state.agentic_trace.append(("📤 Form Agent", f"Submitted {form_label} application for {name}"))
            st.session_state.submitted = True
            st.session_state.form_mode = False  # ✅ Reset to exit form
            st.rerun()


# === RESPONSE EXECUTION ===
if st.session_state.get("pending_response", False):
    latest_query = st.session_state.chat_history[-1]["question"]
    current_role = st.session_state.user_role

    with st.chat_message("assistant"):
        st.markdown("""
        <div class="spinner-container">
            <div class="pulse-ring"></div>
        </div>
        """, unsafe_allow_html=True)

    try:
        payload = {
            "query": latest_query,
            "role": current_role,
            "openai_api_key": st.session_state.get("openai_api_key", "")
        }

        response = requests.post(f"{BACKEND_URL}/query/", json=payload)
        response.raise_for_status()
        response_data = response.json()

        final_response = response_data.get("result", "⚠️ No response received.")
        st.session_state.agentic_trace = response_data.get("trace", [])
        st.session_state.chat_history[-1]["answer"] = final_response
        st.session_state.pending_response = False

    except requests.exceptions.RequestException as e:
        error_msg = f"⚠️ An error occurred while communicating with the backend: {e}"
        st.session_state.agentic_trace.append(("🚨 Error", str(e)))
        st.session_state.chat_history[-1]["answer"] = error_msg
        st.session_state.pending_response = False

    st.rerun()

# === SIDEBAR ===
with st.sidebar:
    # st.markdown("### 🔐 OpenAI API Key")
    # api_key_input = st.text_input("Enter your API key", type="password")
    # if api_key_input:
    #     st.session_state.openai_api_key = api_key_input.strip()
    #     st.success("API key saved.")
    
    # st.markdown("---")
    st.markdown("### 🧠 Agentic AI Reasoning Trace")
    trace = st.session_state.get("agentic_trace", [])
    if trace:
        for label, value in trace:
            st.markdown(f"""
                <div style="margin-bottom:0.6rem; padding:0.7rem 1rem; background:#ffffff; border-left: 4px solid #007a5e; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    <strong style="color:#007a5e;">{label}</strong><br>
                    <span style="color:#333;">{value}</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No trace yet. Ask a question to see reasoning.")

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    user_role = st.selectbox("You are a:", ["Individual", "Employer", "Employment Agency"], index=["Individual", "Employer", "Employment Agency"].index(st.session_state.user_role))
    if user_role != st.session_state.user_role:
        st.session_state.user_role = user_role
        st.info(f"Role set to: **{user_role}**. New queries will be processed with this role.")

# === FOOTER ===
st.markdown("""
<div class="footer">
    <span style="color: #6c757d; font-size: 0.9rem;">
        ⚠️ Prototype by Virtusa using MOM.gov.sg public data. Not affiliated with MOM. Verify with
        <a href="https://www.mom.gov.sg" target="_blank">official sources</a>.
    </span>
</div>
""", unsafe_allow_html=True)