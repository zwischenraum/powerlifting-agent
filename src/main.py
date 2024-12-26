import re
from os import getenv
from openai import OpenAI
from logger import setup_logger
from swarm import Swarm, Agent
from rules_search import search_rules

from dotenv import load_dotenv
from swarm.repl import run_demo_loop

load_dotenv()

CHAT_INSTRUCTIONS = "You are a helpful agent that gives friendly and helpful answers to a fellow powerlifter."

ROUTER_INSTRUCTIONS = """You are a helpful agent that distributes user requests to the appropriate agent.

If a user wants to search openpowerlifting.org, or wants to know any historical records please redirect them to the search agent.

If a user just wants to chat about powerlifting, please redirect him to the chat agent.

If a user wants to know more about a particular powerlifting rule, please redirect him to the rule agent."""

SEARCH_INSTRUCTIONS = """You are a helpful search agent, that helps lifters search openpowerlifting.org. If a user wants to know any (historical) records of a lifter, please search on openpowerlifting.org.

Always ask for and use the lifter's full, official name as it appears on openpowerlifting.org. Common nicknames or shortened names won't work (e.g., searching for 'Mike Tuchscherer' won't find results for 'Michael Tuchscherer')."""

RULES_INSTRUCTIONS = """You are a helpful agent that assists with powerlifting rule questions.
You can search the IPF rulebook to find relevant rules and explain them in a clear way.
Always use the search_rules function to find relevant rules before answering questions."""


def search_openpowerlifting(lifter_name: str) -> str:
    """Search openpowerlifting.org for a lifter's competition history.

    Args:
        lifter_name: The name of the lifter to search for. Should match their name
                    as it appears on openpowerlifting.org.

    Returns:
        str: The lifter's competition history in CSV format if found,
             or an error message if the search fails.

    Note:
        This function first searches the lifter page to find their unique username,
        then uses that to fetch their detailed competition history via the API.
    """
    from requests import get, RequestException

    try:
        # Search for lifter's username
        url = f"http://openpowerlifting.org/lifters.html?q={lifter_name}"
        response = get(url, timeout=10)
        response.raise_for_status()

        match = re.search(r'/api/liftercsv/([^"]+)', response.text)
        if not match:
            return f"Could not find lifter '{lifter_name}' on OpenPowerlifting.org"

        # Fetch competition history
        username = match.group(1)
        csv_url = f"http://openpowerlifting.org/api/liftercsv/{username}"
        csv_response = get(csv_url, timeout=10)
        csv_response.raise_for_status()
        
        return csv_response.text

    except RequestException as e:
        return f"Error accessing OpenPowerlifting.org: {str(e)}"
    except Exception as e:
        return f"Unexpected error while searching: {str(e)}"


def main():
    # Initialize logging
    logger = setup_logger()
    logger.debug("Starting powerlifting agent application")
    
    openai_client = OpenAI(api_key=getenv('OPENAI_API_KEY'), base_url=getenv('OPENAI_BASE_URL'))
    swarm = Swarm(openai_client)

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
        functions=[redirect_to_router_agent, search_rules]
    )

    response = swarm.run(agent=router, messages=[{'role':'user', 'content':'What is a valid bench?'}])
    print(response)
    # run_demo_loop(starting_agent=router, debug=True)


if __name__ == '__main__':
    main()
