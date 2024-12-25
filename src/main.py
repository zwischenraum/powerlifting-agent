import re
from os import getenv

from openai import OpenAI
from swarm import Swarm, Agent

from dotenv import load_dotenv
from swarm.repl import run_demo_loop

load_dotenv()

CHAT_INSTRUCTIONS = "You are a helpful agent that gives friendly and helpful answers to a fellow powerlifter."

ROUTER_INSTRUCTIONS = """You are a helpful agent that distributes user requests to the appropriate agent.

If a user wants to search openpowerlifting.org, or wants to know any historical records please redirect them to the search agent.

If a user just wants to chat about powerlifting, please redirect him to the chat agent.

If a user wants to know more about a particular powerlifting rule, please redirect him to the rule agent."""

SEARCH_INSTRUCTIONS = """You are a helpful search agent, that helps lifters search openpowerlifting.org. If a user wants to know any (historical) records of a lifter, please search on openpowerlifting.org."""

RULES_INSTRUCTIONS = ""


def search_openpowerlifting(lifter_name: str) -> str:
    """Search openpowerlifting.org for a lifter.

    lifter_name: The name of the lifter to search for."""
    from requests import get

    url = f"http://openpowerlifting.org/lifters.html?q={lifter_name}"
    response = get(url)

    match = re.search(r'/api/liftercsv/([^"]+)', response.text)

    if match:
        username = match.group(1)
        csv_response = get(f"http://openpowerlifting.org/api/liftercsv/{username}")
        return csv_response.text
    else:
        return "There was an error searching openpowerlifting.org. Please try again."


def main():
    openai_client = OpenAI(api_key=getenv('OPENAI_API_KEY'), base_url=getenv('OPENAI_BASE_URL'))
    client = Swarm(openai_client)

    def redirect_to_router_agent():
        """Call this function if a user is asking about a topic that is not handled by the current agent."""
        return router

    def redirect_to_search_agent():
        return search

    def redirect_to_chat_agent():
        return chat

    def redirect_to_rule_agent():
        return rule

    router = Agent(
        name="Router Agent",
        # model="llama-3.1-405b-fp8",
        instructions=ROUTER_INSTRUCTIONS,
        functions=[redirect_to_search_agent, redirect_to_chat_agent, redirect_to_rule_agent]
    )

    search = Agent(
        name="Search Agent",
        instructions=SEARCH_INSTRUCTIONS,
        functions=[redirect_to_router_agent, search_openpowerlifting]
    )

    chat = Agent(
        name="Chat Agent",
        instructions=CHAT_INSTRUCTIONS,
        functions=[redirect_to_router_agent]
    )

    rule = Agent(
        name="Rules Agent",
        instructions=RULES_INSTRUCTIONS,
        functions=[redirect_to_router_agent]
    )

    run_demo_loop(starting_agent=router, debug=True)


if __name__ == '__main__':
    main()
