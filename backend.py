import os
from datetime import datetime
from typing import List, Optional, TypedDict, NamedTuple
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote, quote_plus

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

os.environ["OPENAI_API_KEY"] = ""

INDEX_PATH = "faiss_index"

# === LOGGING ===
logger = logging.getLogger("agentic_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# === STATE ===
class AgentState(TypedDict):
    input: str
    role: str
    context: Optional[str]
    agent_output: Optional[str]
    rag_output: Optional[str]
    source: Optional[str]
    reference: Optional[str]
    tool_output: Optional[str]
    result: Optional[str]
    logger: Optional[logging.Logger]
    trace: Optional[List[tuple]]

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

# === RAG RETRIEVAL NODE ===
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

# === RAG GENERATION NODE ===
def rag_generation(state):
    trace = state.get("trace", [])
    logger = state.get("logger")
    query = state["input"]
    context = state.get("context", "")
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)

    # Step 1: FAISS RAG
    # trace.append(("📚 RAG Agent", "Trying FAISS-based RAG generation..."))
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: x["context"],
            question=lambda x: x["input"],
            role=lambda x: x["role"]
        ) | qa_prompt | llm | StrOutputParser()
    )
    result = chain.invoke(state)
    logger.info(f"FAISS RAG result: {result}")

    if "I'm sorry, I couldn't find specific" not in result:
        # trace.append(("📚 RAG Agent", "Used FAISS context to generate answer."))
        return {
            "rag_output": result,
            "source": "rag",
            "reference": "FAISS Vectorstore",
            "trace": trace
        }

    # Step 2: MOM Website via DuckDuckGo
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://html.duckduckgo.com/html/?q=site:mom.gov.sg+{quote_plus(query)}"
        html = requests.get(search_url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        links = soup.select(".result__title a")

        mom_links = []
        for a in links:
            href = a.get("href", "")
            if "uddg=" in href:
                parsed = parse_qs(urlparse(href).query)
                real_url = parsed.get("uddg", [""])[0]
                if real_url.startswith("https://www.mom.gov.sg"):
                    mom_links.append((a.get_text(strip=True), real_url))

        logger.info(f"MOM links found: {mom_links}")

        if mom_links:
            top_title, top_url = mom_links[0]
            page = requests.get(top_url, headers=headers, timeout=10)
            page_soup = BeautifulSoup(page.text, "html.parser")
            paragraphs = page_soup.find_all(["p", "li"])
            raw_text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)
            context = raw_text[:3000]

            # Rerun summarization with LLM
            chain = (
                RunnablePassthrough.assign(
                    context=lambda x: context,
                    question=lambda x: state["input"],
                    role=lambda x: state["role"]
                ) | qa_prompt | llm | StrOutputParser()
            )
            result = chain.invoke(state)
            trace.append(("🌐 MOM Website", f"Used content from: {top_url}"))

            # Optional: Final fallback if response is still not useful
            if "I'm sorry" in result:
                result = llm.invoke(f"Please help answer this Singapore manpower-related query: {query}").content

                return {
                    "rag_output": result,
                    "source": "llm",
                    "reference": "LLM fallback after web",
                    "trace": trace
                }

            return {
                "rag_output": result + f"\n\n🔗 **Reference:** [{top_title}]({top_url})",
                "source": "rag",
                "reference": top_url,
                "trace": trace
            }

        else:
            trace.append(("🌐 MOM Website", "No relevant pages found."))

    except Exception as e:
        trace.append(("🌐 MOM Website", f"Error during MOM search: {str(e)}"))

    # === Step 3: Direct LLM Fallback ===
    result = llm.invoke(f"Please help answer this Singapore manpower-related query: {query}").content
    return {
        "rag_output": result,
        "source": "llm",
        "reference": "LLM fallback",
        "trace": trace
    }


# === FINAL RESPONSE ===
def format_bot_response(content: str, title: str = "📌 MOM Assistant", highlight_color="#007a5e") -> str:
    html_content = content.replace("**", "<strong>").replace("<strong>", "</strong>", 1)
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
    response = state.get("rag_output")
    source = state.get("source")

    if response:
        label = "📚 RAG Agent"
        trace.append((label, "Final answer generated."))
        trace.append(("✅ Final Response", f"Delivered at {timestamp}"))
        formatted = format_bot_response(response)
        return {**state, "result": formatted, "trace": trace}

    trace.append(("❌ Error", "No valid output found in any step"))
    return {
        **state,
        "result": format_bot_response("⚠️ Sorry, I couldn't find an answer."),
        "trace": trace
    }

# === GRAPH CONSTRUCTION ===
def get_langgraph_agent(logger):
    builder = StateGraph(AgentState)
    builder.add_node("rag_retrieval", rag_retrieval)
    builder.add_node("rag_generation", rag_generation)
    builder.add_node("generate_final_answer", generate_final_answer)
    builder.set_entry_point("rag_retrieval")
    builder.add_edge("rag_retrieval", "rag_generation")
    builder.add_edge("rag_generation", "generate_final_answer")
    return builder.compile()

class LangGraphAgentWrapper:
    def __init__(self, logger):
        self.graph = get_langgraph_agent(logger)

    def run(self, query, role, logger):
        return self.graph.invoke({
            "input": query,
            "role": role,
            "logger": logger,
            "trace": [("🧑‍💬 User Message", query)]
        })


# === MAIN EXECUTION TEST ===
if __name__ == "__main__":
    from pprint import pprint

    # Sample prompt to simulate
    sample_query = "Can a domestic worker change employers in Singapore?"
    sample_role = "Individual"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    wrapper = LangGraphAgentWrapper(logger)
    result = wrapper.run(query=sample_query, role=sample_role, logger=logger)
    print("\n===== FINAL OUTPUT =====")
    pprint(result["result"])
    print("\n===== AGENTIC TRACE =====")
    for label, step in result["trace"]:
        print(f"{label}: {step}")