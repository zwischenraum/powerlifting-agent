import logging
from os import getenv

from agent_setup import setup_agents
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from swarm import Swarm

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
agents = setup_agents()


class ChatRequest(BaseModel):
    agent_name: str
    messages: list


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes messages and returns responses from the appropriate agent.
    """
    try:
        logging.info(
            f"Received chat request for agent '{request.agent_name}' with {len(request.messages)} messages"
        )

        if request.agent_name not in agents:
            raise HTTPException(
                status_code=400, detail=f"Unknown agent: {request.agent_name}"
            )

        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Get response from swarm
        response = swarm.run(
            agent=agents[request.agent_name], messages=request.messages
        )
        logging.info(f"Agent '{response.agent.name}' generated response")

        return ChatRequest(agent_name=response.agent.name, messages=response.messages)

    except HTTPException:
        raise
    except KeyError as e:
        logging.error(f"Invalid agent or configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logging.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}
