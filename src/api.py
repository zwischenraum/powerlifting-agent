from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from os import getenv
from swarm import Swarm
from main import setup_agents
from dotenv import load_dotenv
import logging
from logger import setup_logger

# Setup logging
logger = setup_logger()

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Powerlifting Assistant API",
    description="API for interacting with the powerlifting assistant chatbot",
    version="1.0.0"
)

# Initialize OpenAI client and agents
openai_client = OpenAI(api_key=getenv('OPENAI_API_KEY'), base_url=getenv('OPENAI_BASE_URL'))
swarm = Swarm(openai_client)
router = setup_agents()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str
    agent_name: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes messages and returns responses from the appropriate agent.
    """
    try:
        logger.debug(f"Received chat request with {len(request.messages)} messages")
        
        # Convert Pydantic messages to dict format expected by swarm
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get response from swarm
        response = swarm.run(agent=router, messages=messages)
        
        return ChatResponse(
            response=response.content,
            agent_name=response.agent_name if hasattr(response, 'agent_name') else "Unknown"
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
