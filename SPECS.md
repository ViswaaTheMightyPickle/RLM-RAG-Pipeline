# MVP Specification: Hierarchical Recursive RAG (HR-RAG)
**Version:** 1.0  
**Concept:** A "Worker-Orchestrator" pipeline that uses a small model as a semantic filter to maximize the reasoning efficiency of a larger root model.

---

## 1. System Overview
The HR-RAG system bypasses the limitations of standard vector retrieval by using a **Worker Model (0.5B - 1B)** to distill raw text into high-density summaries. These summaries are then gated through a **Probabilistic Voting** mechanism before being presented to the **Root Model** for final reasoning.

---

## 2. Component Architecture

### A. The Root Model (The Orchestrator)
* **Role:** Reasoning, query decomposition, and final synthesis.
* **Key Tasks:**
    * Generate a `core_question` based on the user prompt.
    * Select the appropriate **Persona** for the Worker Model.
    * Make the final decision to "pull" data from the discarded JSON log if the primary context is insufficient.

### B. The Worker Model (The Distiller & Gater)
* **Role:** High-speed text processing and binary classification.
* **Key Tasks:**
    * **Distillation:** Summarize ~4000 token clusters into ~500 token "semantic nuggets."
    * **Gating:** Perform a 1-pass relevance check.
    * **Voting:** Perform $N$ iterations of a "Yes/No" check to generate a relevance percentage.

### C. Data Management
* **Context Window:** The active memory of the Root Model.
* **Discarded Log (`discarded_context.json`):** A persistent JSON file storing rejected summaries, metadata, and their probabilistic scores.

---

## 3. The Processing Pipeline

### Phase I: Retrieval & Distillation
1.  **Ingestion:** RAG identifies relevant document indices/clusters.
2.  **Persona Assignment:** Root Model assigns a persona (e.g., *Technical, Legal, Narrative*) based on initial document metadata.
3.  **Summarization:** Worker Model processes each cluster using the assigned persona.

### Phase II: The Gating Mechanism
For each summary, the Worker Model evaluates its relevance to the `core_question`.
* **Pass 1:** If Worker returns "YES," the summary is added to the **Root Context**.
* **Pass 2 (Failure/Uncertainty):** If the Root Model determines it cannot answer the query, it triggers a **Voting Loop** on the discarded items.

### Phase III: Probabilistic Scoring
To handle "grey area" relevance, the system calculates a score $S$:

$$S = \left( \frac{\sum_{i=1}^{n} v_i}{n} \right) \times 100$$

Where:
* $n$ = number of iterations (e.g., 3 or 5).
* $v_i$ = the binary vote (1 for YES, 0 for NO) from the Worker Model during iteration $i$.

**Threshold Logic:** If $S \ge \text{Threshold}$ (e.g., 70%), the summary is promoted from the JSON log back into the Root Model's context.

---

## 4. Technical Stack
| Component | Technology |
| :--- | :--- |
| **Inference Engine** | LM Studio (OpenAI-compatible REST API) |
| **Language** | Pure Python 3.10+ |
| **Communication** | `requests` library |
| **Storage** | Standard File System (JSON) |
| **Worker Model** | 0.5B - 1.5B Parameter LLM (e.g., Qwen2.5-0.5B or Phi-3.5-mini) |
| **Root Model** | 7B - 70B Parameter LLM (e.g., Llama 3.1 or Mistral) |

---

## 5. Data Schema (`discarded_context.json`)
Each rejected entry must be stored with enough metadata to allow the Root Model to "re-discover" it.

```json
{
  "timestamp": "ISO-8601",
  "root_query": "The original user prompt",
  "rejected_clusters": [
    {
      "cluster_id": "int",
      "source_pages": "string",
      "persona_used": "string",
      "summary": "The 500-token distilled text",
      "relevance_score": "float (0.0 - 1.0)",
      "iteration_count": "int"
    }
  ]
}
```

---

## 6. MVP Success Criteria
* **Compression Ratio:** Successfully reduce raw context by at least 80% while maintaining answer accuracy.
* **Latency:** The Worker Model’s distillation and gating must happen in parallel or fast enough to avoid user-experience lag.
* **Recovery:** The Root Model must demonstrate the ability to "backtrack" and successfully answer a query using a summary retrieved from the `discarded_context.json`.

This technical blueprint outlines a pure-Python implementation of the **HR-RAG** architecture. By keeping the stack lightweight and relying on LM Studio’s OpenAI-compatible API, you maintain maximum portability and low overhead.

## 1. The Tech Stack

| Component | Library/Tool | Reason |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Standard for AI/LLM orchestration. |
| **API Client** | `requests` | Direct control over LM Studio’s REST headers. |
| **Vector DB** | `ChromaDB` or `FAISS` | Lightweight, local-first vector retrieval for initial RAG. |
| **Data Models** | `Pydantic` | Ensures consistent JSON structures for summaries. |
| **CLI** | `Click` | Clean, decorator-based CLI implementation. |
| **Environment** | `python-dotenv` | For managing base URLs and model IDs. |

---

## 2. Hypothetical Codebase Structure

A modular structure allows you to swap out models or vector databases without breaking the core logic.

```text
hr_rag_project/
├── data/                  # Persistent storage for JSON logs
├── src/
│   ├── __init__.py
│   ├── client.py          # LM Studio API wrapper
│   ├── prompts.py         # System prompt library (Personas)
│   ├── rag_engine.py      # Vector search and chunking
│   ├── processor.py       # Distillation, Gating, and Voting logic
│   └── orchestrator.py    # The "Brain" - coordinating the flow
├── main.py                # CLI Entry point
└── .env                   # Configuration (Ports, Model names)
```

---

## 3. Core Functionality Implementation

### A. API Interaction (`client.py`)
Since LM Studio mimics the OpenAI API, you can point to `localhost:1234` easily.

```python
import requests

class LLMClient:
    def __init__(self, base_url="http://localhost:1234/v1"):
        self.url = f"{base_url}/chat/completions"

    def generate(self, model_id, system_msg, user_msg, temp=0.3):
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": temp
        }
        response = requests.post(self.url, json=payload)
        return response.json()['choices'][0]['message']['content']
```

### B. Distillation & Gating Logic (`processor.py`)
This is the "meat" of your idea—the worker model acts as a filter.

```python
def distill_and_gate(worker, cluster, core_question, persona):
    # 1. Distill
    summary = worker.generate("worker-model", persona, f"Summarize: {cluster}")
    
    # 2. Gate (Binary check)
    gate_prompt = "Answer only 'YES' or 'NO'. Does this help answer the question?"
    vote = worker.generate("worker-model", gate_prompt, f"Q: {core_question}\nText: {summary}")
    
    is_relevant = "YES" in vote.upper()
    return summary, is_relevant

def voting_loop(worker, summary, core_question, iterations=3):
    yes_count = 0
    for _ in range(iterations):
        vote = worker.generate("worker-model", "Persona: Voter", f"Q: {core_question}\nText: {summary}")
        if "YES" in vote.upper():
            yes_count += 1
    return yes_count / iterations  # Probabilistic Score
```

---

## 4. The CLI Interface (`main.py`)
Using `Click`, you can create a tool that allows you to specify the model and query directly from the terminal.

```python
import click
from src.orchestrator import HR_Orchestrator

@click.command()
@click.argument('query')
@click.option('--threshold', default=0.7, help='Voting threshold for discarded logs.')
@click.option('--persona', default='TECHNICAL', help='The worker persona to use.')
def run(query, threshold, persona):
    """Hierarchical Recursive RAG CLI"""
    click.echo(f"Processing query: {query}...")
    orchestrator = HR_Orchestrator(threshold=threshold, persona=persona)
    answer = orchestrator.execute(query)
    click.secho("\nFinal Answer:", fg='green', bold=True)
    click.echo(answer)

if __name__ == '__main__':
    run()
```

---

## 5. Connecting to the APIs

To make this work with LM Studio:
1.  **Launch LM Studio:** Load your models (e.g., Llama-3-8B as Root, Qwen-0.5B as Worker).
2.  **Enable Local Server:** Click the `<->` icon on the left sidebar.
3.  **Cross-Port Configuration:** * If running **one** model: Both Root and Worker use `localhost:1234`. The code just swaps the `model` string in the JSON payload.
    * If running **two** models: Open two instances of LM Studio (or one LM Studio and one Ollama) and point the `LLMClient` to their respective ports (e.g., `1234` and `11434`).

---

## 6. The Backtracking Workflow
The "Working Solution" relies on a conditional loop in the orchestrator:

1.  **Try Phase:** Run the distillation. If 1-pass Gating is "YES," send to Root.
2.  **Verify Phase:** If Root responds with a "low confidence" token or explicit "I don't know," trigger the **Backtracking**.
3.  **Recovery Phase:** * Load `discarded_context.json`.
    * Run `voting_loop()` on all entries.
    * Re-calculate the relevance score $S$:
        $$S = \frac{\text{sum of positive votes}}{\text{total iterations}}$$
    * Inject entries where $S \ge \text{threshold}$ back into the prompt and re-run the Root model.

---

### Summary of Implementation Actions
1.  **Implement `rag_engine.py`** to handle PDF loading and initial vector retrieval.
2.  **Implement `processor.py`** to handle the worker's persona-based distillation.
3.  **Develop the JSON Logger** to track discarded clusters with their calculated $S$ scores.
4.  **Wire the Orchestrator** to handle the iterative "Root check -> Discard scan -> Final answer" loop.


github email: thereddragonspeaks22919@protonmail.com
github username: ViswaaTheMightyPickle
github url: https://github.com/ViswaaTheMightyPickle/RLM-RAG-Pipeline.git
