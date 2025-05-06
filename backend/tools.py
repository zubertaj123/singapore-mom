from datetime import datetime
from langchain.tools import Tool

def log_tool_usage(tool_logs, tool_name, user_input):
    tool_logs.append({
        "tool": tool_name,
        "input": user_input,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def dependent_pass_checker(_, logs): 
    log_tool_usage(logs, "Dependant Pass Tool", _)
    return "🧑‍👩‍👧‍👦 Requires salary ≥ SGD 6,000/month for Dependant Pass eligibility."

def quota_checker(_, logs): 
    log_tool_usage(logs, "Quota Tool", _)
    return "📊 S Pass quota is capped at 10% for services sector."

def license_checker(_, logs): 
    log_tool_usage(logs, "License Tool", _)
    return "📜 EA must have valid MOM license: https://www.mom.gov.sg/eservices/services/employment-agency-licence"

def appointment_api_tool(input_text, logs): 
    log_tool_usage(logs, "Appointment Tool", input_text)
    return "📅 Appointment scheduled (simulated). Confirmation sent."

def wpol_application_tool(input_text, logs): 
    log_tool_usage(logs, "WPOL Submission Tool", input_text)
    return "📂 Work Permit Application submitted (simulated). Tracking ID: WPOL-23918712."

def status_api_tool(input_text, logs): 
    log_tool_usage(logs, "Status Checker Tool", input_text)
    return "📦 Application is in queue. Response in 3–5 business days."
