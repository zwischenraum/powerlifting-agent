from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from os import getenv
from swarm import Agent, Swarm
from agent_setup import setup_agents
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Powerlifting Assistant API",
    description="API for interacting with the powerlifting assistant chatbot",
    version="1.0.0",
)

# Initialize OpenAI client and agents
openai_client = OpenAI(
    api_key=getenv("OPENAI_API_KEY"), base_url=getenv("OPENAI_BASE_URL")
)
swarm = Swarm(openai_client)
router = setup_agents()


class ChatRequest(BaseModel):
    agent: Agent
    messages: list


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes messages and returns responses from the appropriate agent.
    """
    try:
        logging.debug(f"Received chat request with {len(request.messages)} messages")

        # Get response from swarm
        response = swarm.run(agent=request.agent, messages=request.messages)

        return ChatRequest(agent=response.agent, messages=response.messages)
    except Exception as e:
        logging.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}
