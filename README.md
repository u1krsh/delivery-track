---
title: DeliveryAgent
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# 🚀 Delivery Tracker — OpenEnv Benchmark

A **real-world-inspired delivery dispatch benchmark** for evaluating AI agents on multi-driver assignment, graph-based routing, and order-completion tasks with dense reward signals.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Spaces-blue)](https://huggingface.co/spaces)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## 📋 Overview

This environment simulates a **food delivery dispatch system** where an AI agent must:

1. **Assign** delivery orders to available drivers
2. **Route** drivers through a weighted city graph to pickup locations
3. **Pick up** food at restaurants
4. **Navigate** to customer destinations
5. **Complete** deliveries while minimising total travel cost

The agent interacts through the standard OpenEnv `reset()` / `step()` / `state()` API, receiving structured JSON observations and issuing discrete actions. A dense reward function provides signal at every step — not just at episode end.

## 🏗️ Architecture

```
delivery-tracker/
├── env/                          # OpenEnv environment
│   ├── __init__.py               # Package init (exports DeliveryEnvironment)
│   ├── environment.py            # Core env: reset(), step(), state()
│   ├── models.py                 # Pydantic models (Action, Observation, etc.)
│   ├── tasks.py                  # 3 deterministic tasks (easy/medium/hard)
│   └── graders.py                # Graders (scores 0.0–1.0)
├── src/                          # Domain logic (used by env)
│   ├── models/                   # Graph, Driver, Delivery classes
│   ├── algorithms/               # BFS, DFS, Dijkstra, A* routing
│   ├── services/                 # AssignmentService
│   └── gui/                      # Tkinter GUI for the AI agent
├── inference.py                  # Baseline LLM agent (HuggingFace API)
├── entrypoint.py                 # Docker container entry point
├── healthcheck.py                # HTTP health check server
├── run.py                        # GUI launcher
├── openenv.yaml                  # OpenEnv spec metadata
├── Dockerfile                    # Multi-stage production build
├── docker-compose.yml            # Local dev runner
├── requirements.txt              # Python dependencies
└── .env.example                  # Environment variable template
```

## 🎮 Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `assign_driver` | `driver_id`, `delivery_id` | Assign a delivery order to a driver |
| `move_driver` | `driver_id`, `target_node` | Move a driver to an adjacent graph node |
| `pickup_delivery` | `driver_id`, `delivery_id` | Pick up a delivery at the pickup node |
| `complete_delivery` | `driver_id`, `delivery_id` | Complete a delivery at the destination |

## 👁️ Observation Space

Each observation includes:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Current task identifier |
| `task_prompt` | `str` | Natural-language task description |
| `step_number` | `int` | Current step (0-indexed) |
| `max_steps` | `int` | Maximum steps allowed |
| `nodes` | `list[NodeInfo]` | Graph nodes with neighbors and edge weights |
| `drivers` | `list[DriverInfo]` | Driver states (location, assignments, capacity) |
| `deliveries` | `list[DeliveryInfo]` | Delivery states (status, pickup, destination) |
| `available_actions` | `list[str]` | Valid action descriptions |
| `last_action_result` | `str` | Result of the previous action |

## 📊 Tasks

| Task | Difficulty | Graph | Drivers | Deliveries | Max Steps | Key Challenge |
|------|-----------|-------|---------|------------|-----------|---------------|
| **easy** | ⭐ | 4-node chain | 1 | 1 | 10 | Basic routing |
| **medium** | ⭐⭐ | 9-node 3×3 grid | 2 | 3 | 25 | Multi-driver assignment |
| **hard** | ⭐⭐⭐ | 16-node 4×4 grid | 3 | 5 (with pickup) | 50 | Capacity + congestion |

## 🏆 Scoring

Composite score in **[0.0, 1.0]** with four dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Completion | 40% | Fraction of deliveries completed |
| Efficiency | 25% | Optimal travel cost / actual travel cost |
| Speed | 20% | Steps saved relative to max_steps |
| Validity | 15% | 1.0 minus penalty per invalid action |

## 💰 Reward Function

Dense reward at every step:

| Component | Value | Trigger |
|-----------|-------|---------|
| `delivery_progress` | +0.1 | Moving toward delivery goal |
| `pickup_bonus` | +0.2 | Picking up at pickup node |
| `delivery_completion` | +1.0 | Completing a delivery |
| `efficiency_bonus` | 0.0–0.5 | Near-optimal routing on completion |
| `invalid_action_penalty` | −0.3 | Attempting an invalid action |
| `loop_penalty` | −0.1 | Revisiting a seen state |
| `idle_penalty` | −0.05 | Valid action with no progress |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- `pip install -r requirements.txt`

### Run the GUI

```bash
python run.py
```

This launches the AI Agent GUI where you can:
- Select a task (easy / medium / hard)
- Watch the AI agent dispatch deliveries in real-time
- View live action log and composite scores

### Run Inference (CLI)

```bash
# Uses embedded HuggingFace API key by default
python inference.py

# Or override with your own credentials
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export OPENAI_API_KEY=sk-...
python inference.py
```

### Docker

```bash
# Build and run
docker build -t delivery-tracker-openenv .
docker run --rm -p 7860:7860 delivery-tracker-openenv

# With docker-compose
cp .env.example .env
docker compose up --build
```

## 🤗 HuggingFace Spaces Deployment

The environment is designed to run as a containerised HuggingFace Space:

1. Push the repository to a new HF Space
2. Set the Space SDK to "Docker"
3. The `Dockerfile` handles everything: installs deps, runs smoke tests, starts health check + inference
4. Environment variables can be set in Space settings

## 🔧 API Configuration

The inference script works out-of-the-box with embedded HuggingFace credentials:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api-inference.huggingface.co/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.3` | LLM model identifier |
| `HF_TOKEN` | *(embedded)* | HuggingFace API token |
| `OPENAI_API_KEY` | *(none)* | Alternative: OpenAI API key |

## 📈 Baseline Scores

Baseline performance with `Mistral-7B-Instruct-v0.3` via HuggingFace Inference API:

| Task | Score | Completion | Efficiency | Speed | Validity |
|------|-------|------------|------------|-------|----------|
| easy | — | — | — | — | — |
| medium | — | — | — | — | — |
| hard | — | — | — | — | — |

*(Run `python inference.py` to generate baseline scores)*

## 📜 License

MIT License
