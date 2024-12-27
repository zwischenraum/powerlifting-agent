# Powerlifting Agent

An AI-powered assistant for powerlifters that helps with rule interpretations, record lookups, and general powerlifting discussions.

## Features

- **Rule Search**: Intelligent search through IPF rulebook using hybrid search (BM25 + semantic search)
- **Record Lookup**: Integration with OpenPowerlifting.org for competition records and athlete stats
- **Chat Assistant**: General powerlifting discussion and advice
- **Smart Routing**: Automatically directs queries to the most appropriate specialized agent

## Architecture

- Built with Python 3.10+
- Uses Qdrant vector database for semantic search
- Multiple specialized agents powered by LLM for different tasks
- FastAPI backend with Streamlit frontend
- Docker containerization for all components

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- OpenAI API key

## Project Structure

```
.
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── api.py             # FastAPI backend
│   ├── frontend.py        # Streamlit frontend
│   ├── main.py           # Application entry point
│   ├── agent_setup.py    # Agent configuration
│   └── rules_search.py   # IPF rulebook search implementation
├── data/                  # Data files
│   └── rulebook.txt      # IPF Technical Rules
└── docker/               # Docker configuration
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd powerlifting-agent
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables in `.env`:
```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=your-base-url  # Optional, if using a different endpoint
```

## Running the Application

Start all services using Docker Compose:
```bash
docker compose up --build
```

This will start:
- Qdrant vector database on port 6333
- FastAPI backend on port 8000
- Streamlit frontend on port 8501

## Development

The project uses Poetry for dependency management. To add new dependencies:
```bash
poetry add package-name
```

### Code Organization

- `api.py`: FastAPI application with chat endpoints
- `frontend.py`: Streamlit web interface
- `agent_setup.py`: Configuration for different specialized agents
- `rules_search.py`: Hybrid search implementation for IPF rulebook

### Docker Configuration

- `Dockerfile.server`: Backend service configuration
- `Dockerfile.frontend`: Frontend service configuration
- `docker-compose.yml`: Multi-container orchestration

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
