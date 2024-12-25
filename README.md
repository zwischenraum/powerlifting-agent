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
- Docker support for Qdrant deployment

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- OpenAI API key

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

3. Start Qdrant:
```bash
docker-compose up -d
```

4. Set up environment variables:
```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=your-base-url  # Optional, if using a different endpoint
```

## Usage

Run the main application:
```bash
poetry run python src/main.py
```

The agent will start in interactive mode where you can:
- Ask questions about powerlifting rules
- Look up competition records
- Get general powerlifting advice

## Development

The project uses Poetry for dependency management. To add new dependencies:
```bash
poetry add package-name
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
