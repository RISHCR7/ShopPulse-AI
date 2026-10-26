"""Optional local GenAI summaries via Ollama HTTP API. Requires Ollama running locally."""
import requests


def summarize_business_metrics(metrics: dict, model: str = "llama3.2") -> str:
    prompt = ("You are an e-commerce operations analyst. Based only on these metrics, "
              "write a concise executive summary, 3 observations, and 3 cautious actions. "
              "Do not invent numbers or claim causality. Metrics: " + str(metrics))
    response = requests.post("http://localhost:11434/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    response.raise_for_status()
    return response.json()["response"]
