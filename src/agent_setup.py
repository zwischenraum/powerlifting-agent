from rules_search import search_rules
from swarm import Agent

CHAT_INSTRUCTIONS = """
You are the chat agent - a helpful agent that gives friendly and helpful answers to a fellow
powerlifter.
"""

ROUTER_INSTRUCTIONS = """
You are the router agent - a helpful agent that distributes user requests to the appropriate agent.
If a user wants to search openpowerlifting.org, or wants to know any historical records please
redirect them to the search agent.
If a user just wants to chat about powerlifting, please redirect them to the chat agent.
If a user wants to know more about a particular powerlifting rule, please redirect them to the
rule agent.
"""

SEARCH_INSTRUCTIONS = """
You are the search agent - a helpful agent, that helps lifters search openpowerlifting.org.
If a user wants to know any (historical) records of a lifter, please search on openpowerlifting.org.

Always ask for and use the lifter's full, official name as it appears on openpowerlifting.org.
Common nicknames or shortened names won't work (e.g., searching for 'Mike Tuchscherer' won't find
results for 'Michael Tuchscherer').
"""

RULES_INSTRUCTIONS = """
You are the rule agent - a helpful agent that assists with powerlifting rule questions.
You can search the IPF rulebook to find relevant rules and explain them in a clear way.
Always use the search_rules function to find relevant rules before answering questions.
"""


def search_openpowerlifting(lifter_name: str) -> str:
    """Search openpowerlifting.org for a lifter's competition history."""
    import re

    from requests import RequestException, get

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


def setup_agents() -> dict[str, Agent]:
    def redirect_to_router_agent():
        """Call this function if a user is asking about a topic that is not handled by the current agent."""
        return router

    def redirect_to_search_agent():
        return search

    def redirect_to_chat_agent():
        return chat

    def redirect_to_rules_agent():
        return rules

    router = Agent(
        name="router",
        instructions=ROUTER_INSTRUCTIONS,
        functions=[
            redirect_to_search_agent,
            redirect_to_chat_agent,
            redirect_to_rules_agent,
        ],
    )

    search = Agent(
        name="search",
        instructions=SEARCH_INSTRUCTIONS,
        functions=[redirect_to_router_agent, search_openpowerlifting],
    )

    chat = Agent(
        name="chat",
        instructions=CHAT_INSTRUCTIONS,
        functions=[redirect_to_router_agent],
    )

    rules = Agent(
        name="rules",
        instructions=RULES_INSTRUCTIONS,
        functions=[redirect_to_router_agent, search_rules],
    )

    return {"router": router, "search": search, "chat": chat, "rules": rules}
