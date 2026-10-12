# Jev tutorial: a ShopEasy support bot

A notebook that walks through the three output types of [TypeSafe AI's Jev](https://typesafe.ai) by building a support bot for ShopEasy, a fictional online store:

- **Choice**: route support tickets to the right team
- **Score**: rerank help articles in a RAG pipeline
- **Noul**: block prompt attacks before they reach the LLM

It finishes with a speed, cost and accuracy benchmark of the Jev guardrail against an OpenAI model with structured output.

## Run it

1. Install dependencies: `uv sync`
2. Add your `TYPESAFE_API_KEY` and `OPENAI_API_KEY` to `.env`
3. Open `jev_tutorial.ipynb` in VS Code and pick the `.venv` kernel, or run `uv run --with jupyter jupyter lab`

## Files

| Path | What's in it |
|---|---|
| `jev_tutorial.ipynb` | The tutorial |
| `data/tickets.csv` | 12 support tickets for the router |
| `data/help_articles.json` | 31 ShopEasy help articles for RAG |
| `data/attacks.csv` | 100 labeled messages for the benchmark (25 per attack type + 25 harmless) |
| `helpers/bench.py` | Benchmark timing, cost, accuracy and charts, plus model prices |
