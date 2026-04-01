# HR-RAG: Hierarchical Recursive RAG Pipeline

A **Worker-Orchestrator** pipeline that uses a small model as a semantic filter to maximize the reasoning efficiency of a larger root model.

## Overview

HR-RAG bypasses the limitations of standard vector retrieval by using a **Worker Model (0.5B-1.5B)** to distill raw text into high-density summaries. These summaries are then gated through a **Probabilistic Voting** mechanism before being presented to the **Root Model** for final reasoning.

## Features

- **Hierarchical Processing**: Worker model distills ~4000 token clusters into ~500 token summaries
- **Probabilistic Voting**: N-iteration relevance scoring for "grey area" content
- **Backtracking Recovery**: Root model can "pull" data from discarded context log
- **Multiple Personas**: Technical, Legal, Narrative, Medical, Financial, General
- **Pure Python**: Lightweight implementation using LM Studio's OpenAI-compatible API

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` to match your LM Studio setup:
   ```env
   LM_STUDIO_BASE_URL=http://localhost:1234/v1
   WORKER_MODEL_ID=qwen2.5-0.5b
   ROOT_MODEL_ID=llama-3.1-8b
   ```

## Usage

### Start LM Studio

1. Open LM Studio
2. Load your Worker model (e.g., Qwen2.5-0.5B)
3. Enable Local Server (click the `<->` icon)
4. Note: You can swap models during execution, or run two instances on different ports

### CLI Commands

**Check health:**
```bash
python main.py health
```

**Ingest documents:**
```bash
python main.py ingest document.txt
python main.py ingest manual.pdf --source "Product Manual v2.0"
```

**Ask questions:**
```bash
python main.py ask "What is the architecture of the system?"
python main.py ask "Explain the gating mechanism" --persona TECHNICAL
python main.py ask "What are the legal requirements?" --threshold 0.5 --verbose
```

**View statistics:**
```bash
python main.py stats
python main.py discarded
```

**Clear data:**
```bash
python main.py clear
python main.py clear_discarded
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Root Model (Orchestrator)                     │
│  - Decompose query to core_question                              │
│  - Select Worker persona                                         │
│  - Synthesize final answer                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Engine (ChromaDB)                         │
│  - Retrieve relevant document chunks                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Worker Model (Processor)                       │
│  - Distill: 4000 tokens → 500 tokens                             │
│  - Gate: Binary YES/NO relevance check                           │
│  - Vote: N iterations for probabilistic score                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
            ┌──────────────┐   ┌──────────────────┐
            │  Relevant    │   │   Discarded      │
            │  Context     │   │   Log (JSON)     │
            └──────────────┘   └──────────────────┘
                    │                   │
                    │                   │ (if LOW_CONFIDENCE)
                    │                   ▼
                    │         ┌──────────────────┐
                    │         │  Voting Loop     │
                    │         │  Recover items   │
                    │         │  S >= threshold  │
                    │         └──────────────────┘
                    │                   │
                    ▼                   ▼
            ┌─────────────────────────────────────┐
            │      Root Model Final Synthesis     │
            └─────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │           Final Answer              │
            └─────────────────────────────────────┘
```

## Project Structure

```
hr_rag_project/
├── data/                  # Persistent storage for JSON logs
├── src/
│   ├── __init__.py
│   ├── client.py          # LM Studio API wrapper
│   ├── prompts.py         # System prompt library (Personas)
│   ├── models.py          # Pydantic data models
│   ├── rag_engine.py      # Vector search and chunking
│   ├── processor.py       # Distillation, Gating, and Voting logic
│   └── orchestrator.py    # The "Brain" - coordinating the flow
├── main.py                # CLI Entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Probabilistic Voting

The system calculates a relevance score $S$:

$$S = \left( \frac{\sum_{i=1}^{n} v_i}{n} \right) \times 100$$

Where:
- $n$ = number of iterations (default: 3)
- $v_i$ = binary vote (1 for YES, 0 for NO)

Items with $S \geq \text{Threshold}$ (default: 70%) can be recovered from the discarded log during backtracking.

## Success Criteria

- **Compression Ratio**: Reduce raw context by 80%+ while maintaining accuracy
- **Latency**: Worker processing happens efficiently
- **Recovery**: Root model can successfully answer using recovered summaries

## Recommended Models

| Role | Model | Size |
|------|-------|------|
| Worker | Qwen2.5-0.5B | 0.5B |
| Worker | Phi-3.5-mini | 3.8B |
| Root | Llama 3.1-8B | 8B |
| Root | Mistral-7B | 7B |
| Root | Qwen2.5-72B | 72B |

## License

MIT
