# Pipecat Server

### Prerequisites

#### Environment

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager installed

#### AI Service API keys
Reach out to Deepak for the API Keys


### Setup

Follow the steps below for running the service locally

1. Clone this repository

2. Configure your API keys:

   Create a `.env` file:

3. Set up a virtual environment and install dependencies

   ```bash
   uv sync
   ```

### Run your bot locally

```bash
uv run bot.py -t daily
```

**Open http://localhost:7860 in your browser** and click `Connect` to start talking to your bot.

> 💡 First run note: The initial startup may take ~20 seconds as Pipecat downloads required models and imports.

🎉 **Success!** Your bot is running locally.