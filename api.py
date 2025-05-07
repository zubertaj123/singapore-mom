# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from back import LangGraphAgentWrapper
import logging
import os

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("FastAPI application started.")

app = FastAPI()
agent_wrapper = LangGraphAgentWrapper(logger=logger)

class QueryRequest(BaseModel):
    query: str
    role: str

class Response(BaseModel):
    result: str
    source: str
    reference: str
    trace: list = None

@app.post("/query/", response_model=Response)
async def process_query(request: QueryRequest):
    logger.info(f"Processing query: {request.query} with role: {request.role}")
    try:
        result = agent_wrapper.run(request.query, request.role, logger=logger) # Pass the logger
        if "Error processing request: Prompt missing required variables" in result.get("result", ""):
            logger.error(f"Langchain agent prompt error: {result['result']}")
            raise HTTPException(status_code=500, detail=result["result"])

        return Response(
            result=result["result"],
            source=result["source"],
            reference=result["reference"],
            trace=result.get("trace", [("LangGraph", "Execution complete")])
        )


    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
