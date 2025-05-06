from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def build_qa_chain(retriever):
    QA_PROMPT = PromptTemplate(
        template="""
You are a MOM assistant responding to a {role}. Use this context:
{context}

Response rules:
- Tailor tone and details to suit a {role}.
- Be precise with numbers/dates (e.g., "$3,150/month")
- Use **bold** for official terms
- Use bullet points for lists
- Use markdown formatting
- Never invent information

Question: {question}
Answer:""",
        input_variables=["context", "question", "role"]
    )

    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT, "document_variable_name": "context"}
    )
