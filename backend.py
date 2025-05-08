

import os
from datetime import datetime
from typing import List, Optional, TypedDict, NamedTuple
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

INDEX_PATH = "faiss_index"

# === LOGGING ===
logger = logging.getLogger("agentic_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# === UTILS ===
tool_logs = []

class ToolInvocation(NamedTuple):
    name: str
    args: dict

def log_tool_usage(tool_name: str, user_input: str, logger):
    log = {
        "tool": tool_name,
        "input": user_input,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    tool_logs.append(log)
    logger.info(f"Tool '{tool_name}' used with input: '{user_input}'")
    return ("🛠️ Tool Agent", f"{tool_name} used with input: '{user_input}'")

def with_logging(name, func):
    def wrapper(state):
        logger = state.get("logger")
        if logger:
            log_tool_usage(name, state["input"], logger)
        return func(state)
    return wrapper

# === TOOLS ===
def get_all_tools():
    return [
        Tool(name="DependantPassTool", func=with_logging("Dependant Pass Tool", lambda state: "Requires salary ≥ SGD 6,000/month for Dependant Pass eligibility."), description="Check eligibility for Dependant Pass."),
        Tool(name="QuotaTool", func=with_logging("Quota Tool", lambda state: "S Pass quota is capped at 10% for services sector."), description="Check S Pass quota limits."),
        Tool(name="LicenseTool", func=with_logging("License Tool", lambda state: "EA must have valid MOM license: https://www.mom.gov.sg/eservices/services/employment-agency-licence"), description="Verify MOM license."),
        Tool(name="AppointmentTool", func=with_logging("Appointment Tool", lambda state: "Appointment scheduled (simulated). Confirmation sent."), description="Simulate MOM appointment."),
        Tool(name="WPOLSubmissionTool", func=with_logging("WPOL Submission Tool", lambda state: "Work Permit Application submitted (simulated)."), description="Simulate Work Permit submission."),
        Tool(name="StatusCheckerTool", func=with_logging("Status Checker Tool", lambda state: "Application is in queue. Response in 3–5 business days."), description="Check application status.")
    ]

# === VECTORSTORE ===
def load_vectorstore(logger):
    embeddings = OpenAIEmbeddings()
    try:
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("Vectorstore loaded.")
        return vectorstore
    except Exception as e:
        logger.error(f"Vectorstore error: {e}")
        return None

# === PROMPT ===
qa_prompt = PromptTemplate(
    input_variables=["context", "question", "role"],
    template="""
You are an assistant for the Singapore Ministry of Manpower (MOM), responding to a {role}.
Use the provided CONTEXT to answer the user's QUESTION.

### CONTEXT:
{context}

### QUESTION:
{question}

### INSTRUCTIONS:
- Use the CONTEXT above to answer the QUESTION.
- It's okay to match similar terms (e.g. "dependent permit" ≈ "Dependant's Pass") if the meaning is clear.
- If the answer is **partially** available, respond with the best possible summary based on what's provided.
- If nothing related is found even loosely, reply:
  "**I'm sorry, I couldn't find specific information on that.**"
- Tailor tone and detail for a {role}.
- Use **bold** for official terms.
- Format lists with bullet points.
- Use markdown formatting.

### ANSWER:
"""
)


# === LANGGRAPH NODES ===
def rag_retrieval(state):
    logger = state.get("logger")
    trace = state.get("trace", [])
    vectorstore = load_vectorstore(logger)
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        documents = retriever.invoke(state["input"])
        context_text = "\n\n".join([doc.page_content for doc in documents])
        trace.append(("📚 RAG Retrieval", f"{len(documents)} documents matched."))
        return {"context": context_text, "trace": trace}
    trace.append(("📚 RAG Retrieval", "No documents matched."))
    return {"context": "", "trace": trace}

# def rag_generation(state):
#     llm = ChatOpenAI(model="gpt-4", temperature=0.2)
#     chain = (
#         RunnablePassthrough.assign(
#             context=lambda x: x["context"],
#             question=lambda x: x["input"],
#             role=lambda x: x["role"]
#         )
#         | qa_prompt
#         | llm
#         | StrOutputParser()
#     )
#     result = chain.invoke(state)
#     trace = state.get("trace", [])
#     trace.append(("🧠 RAG Generator", "Generated answer using retrieved documents."))
#     return {"rag_output": result, "source": "rag", "reference": "FAISS Vectorstore", "trace": trace}

def rag_generation(state):
    import requests
    from bs4 import BeautifulSoup

    llm = ChatOpenAI(model="gpt-4", temperature=0.2)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: x["context"],
            question=lambda x: x["input"],
            role=lambda x: x["role"]
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(state)
    trace = state.get("trace", [])

    # Check if RAG result is empty or fallback message
    if "**I'm sorry, I couldn't find" in result:
        trace.append(("📚 RAG Agent", "FAISS result insufficient, triggering MOM website lookup..."))

        # === Perform live web search over MOM.gov.sg ===
        search_query = state["input"]
        search_url = f"https://www.google.com/search?q=site%3Amom.gov.sg+{search_query.replace(' ', '+')}"

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.select("a[href^='https://www.mom.gov.sg']")

            if links:
                first_link = links[0]["href"]
                page = requests.get(first_link, headers=headers)
                page_soup = BeautifulSoup(page.text, "html.parser")
                paragraphs = page_soup.find_all("p")
                page_text = "\n".join(p.get_text(strip=True) for p in paragraphs[:10])  # limit to 10 paras

                # Re-run LLM on fetched content
                new_prompt = qa_prompt.format(context=page_text, question=state["input"], role=state["role"])
                result = llm.invoke(new_prompt).content

                trace.append(("🌐 MOM Website", f"Reviewed page: [{first_link}]({first_link})"))
                return {
                    "rag_output": result,
                    "source": "web_rag",
                    "reference": first_link,
                    "trace": trace
                }

            else:
                trace.append(("🌐 MOM Website", "No relevant pages found."))
                return {
                    "rag_output": "**I'm sorry, I couldn't find specific information on the MoM official website.**",
                    "source": "web_rag",
                    "reference": "No link",
                    "trace": trace
                }

        except Exception as e:
            trace.append(("🌐 MOM Website", f"Error occurred during web search: {str(e)}"))
            return {
                "rag_output": "**I'm sorry, I couldn't find specific information on the MoM official website.**",
                "source": "web_rag",
                "reference": "Error",
                "trace": trace
            }

    else:
        trace.append(("📚 RAG Agent", "Used FAISS context to generate answer."))
        return {
            "rag_output": result,
            "source": "rag",
            "reference": "FAISS Vectorstore",
            "trace": trace
        }


def call_llm_agent(state):
    tools = get_all_tools()
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    agent_executor = create_react_agent(llm, tools)
    result = agent_executor.invoke({
        "messages": [{"role": "user", "content": state["input"]}]
    })
    trace = state.get("trace", [])
    trace.append(("🧠 LLM Agent", "Fallback to ReAct agent-based reasoning."))
    return {
        "agent_output": result.get("output", result),
        "source": "llm",
        "reference": "LLM agent only",
        "trace": trace
    }

def call_tool(state):
    tool_call = state["tool_calls"][0]
    tool = next(t for t in get_all_tools() if t.name == tool_call.name)
    return {"tool_output": tool.func(state)}

def format_tool_output(state):
    if "tool_output" in state and state["tool_calls"]:
        tool_name = state["tool_calls"][0].name
        return {"agent_output": state["tool_output"], "source": "tool", "reference": tool_name}
    return {"agent_output": "No tool output.", "source": "error", "reference": "unknown"}

def rag_or_llm_router(state, logger):
    context = state.get("context", "")
    if context and len(context.strip()) > 100:
        return "rag_generation"
    return "llm_agent"

def should_continue(state):
    if "tool_calls" in state:
        return {"__next__": "call_tool"}
    return {"__next__": "generate_final_answer"}

def format_bot_response(content: str, title: str = "📌 MOM Assistant", highlight_color="#007a5e") -> str:
    html_content = (
        content.replace("**", "<strong>").replace("<strong>", "</strong>", 1)
    )
    return f"""
    <div style="
        background-color: #ffffff;
        border-left: 5px solid {highlight_color};
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #333;
        font-size: 0.95rem;
        line-height: 1.6;
    ">
        <div style="font-weight: 600; color: {highlight_color}; margin-bottom: 0.4rem;">{title}</div>
        {html_content}
    </div>
    """

def generate_final_answer(state):
    trace = state.get("trace", [])
    timestamp = datetime.now().strftime('%H:%M:%S')

    if "rag_output" in state:
        trace.append(("📚 RAG Agent", "Used FAISS context to generate answer"))
        trace.append(("✅ Final Response", f"Delivered at {timestamp}"))

        formatted = format_bot_response(state["rag_output"])
        return {**state, "result": formatted, "trace": trace}

    elif "agent_output" in state:
        trace.append(("🧠 LLM Agent", "Fallback to tool or open-ended response"))
        trace.append(("✅ Final Response", f"Delivered at {timestamp}"))

        formatted = format_bot_response(str(state["agent_output"]))
        return {**state, "result": formatted, "trace": trace}

    trace.append(("❌ Error", "No valid output found in either path"))
    return {
        **state,
        "result": format_bot_response("⚠️ Sorry, I couldn't find an answer."),
        "trace": trace
    }

# === STATE ===
class AgentState(TypedDict):
    input: str
    role: str
    context: Optional[str]
    tool_calls: Optional[List[ToolInvocation]]
    agent_output: Optional[str]
    rag_output: Optional[str]
    source: Optional[str]
    reference: Optional[str]
    tool_output: Optional[str]
    result: Optional[str]
    logger: Optional[logging.Logger]
    trace: Optional[List[tuple]]

# === GRAPH ===
def get_langgraph_agent(logger):
    builder = StateGraph(AgentState)
    builder.add_node("rag_retrieval", rag_retrieval)
    builder.add_node("rag_generation", rag_generation)
    builder.add_node("llm_agent", call_llm_agent)
    builder.add_node("call_tool", call_tool)
    builder.add_node("format_tool_output", format_tool_output)
    builder.add_node("generate_final_answer", generate_final_answer)

    builder.set_entry_point("rag_retrieval")

    builder.add_conditional_edges(
        "rag_retrieval",
        lambda s: rag_or_llm_router(s, s.get("logger")),
        {
            "rag_generation": "rag_generation",
            "llm_agent": "llm_agent"
        }
    )

    builder.add_edge("rag_generation", "generate_final_answer")
    builder.add_conditional_edges("llm_agent", should_continue, {
        "call_tool": "call_tool",
        "generate_final_answer": "generate_final_answer"
    })
    builder.add_edge("call_tool", "format_tool_output")
    builder.add_edge("format_tool_output", "generate_final_answer")
    return builder.compile()


# === WRAPPER ===
class LangGraphAgentWrapper:
    def __init__(self, logger):
        self.graph = get_langgraph_agent(logger)

    def run(self, query, role, logger, openai_api_key=None):
        tool_logs.clear()

        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        logger.info(f"LangGraphAgentWrapper.run called with query: '{query}', role: '{role}'")
        result = self.graph.invoke({
            "input": query,
            "role": role,
            "logger": logger,
            "trace": [("🧑‍💬 User Message", query)]
        })
        logger.info(f"LangGraphAgentWrapper.run result: {result}")
        return {
            "result": result.get("result", "⚠️ No response found."),
            "source": result.get("source", "unknown"),
            "reference": result.get("reference", "unknown"),
            "trace": result.get("trace", [("LangGraph", "Execution complete")])
        }

__all__ = ["LangGraphAgentWrapper", "tool_logs"]