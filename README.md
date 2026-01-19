# AION - Artificial Intelligence Operating Nexus

A production-ready AGI cognitive architecture that demonstrates genuine artificial general intelligence capabilities through integrated cognitive systems.

## 🧠 Overview

AION is a comprehensive AI operating system with five core cognitive systems:

1. **Deterministic Planning Graph** - Reproducible execution planning with NetworkX
2. **Vector Memory Search** - Semantic memory with FAISS backend
3. **Tool Orchestration** - Intelligent multi-tool coordination
4. **Self-Improvement Loop** - Autonomous evolution and optimization
5. **Visual Cortex** - Computer vision with deep reasoning

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/aion.git
cd aion

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Running the Server

```bash
# Using CLI
aion server --host 0.0.0.0 --port 8000

# Or directly
python -m aion.main --host 0.0.0.0 --port 8000
```

### Using the API

```python
import httpx

# Process a request
response = httpx.post(
    "http://localhost:8000/process",
    json={"query": "What is the capital of France?"}
)
print(response.json())

# Search memory
response = httpx.post(
    "http://localhost:8000/memory/search",
    json={"query": "capitals", "limit": 10}
)
print(response.json())

# Execute a tool
response = httpx.post(
    "http://localhost:8000/tools/execute",
    json={
        "tool_name": "calculator",
        "params": {"expression": "2 + 2"}
    }
)
print(response.json())
```

## 📚 Core Systems

### 1. Planning Graph

The planning system uses directed acyclic graphs (DAGs) for deterministic execution:

- Multi-step workflow planning
- Dependency resolution and parallel execution
- Checkpoint/rollback capabilities
- Visual plan representation

```python
from aion.systems.planning import PlanningGraph

graph = PlanningGraph()
await graph.initialize()

# Create a plan
plan = graph.create_plan(name="Data Analysis", description="Analyze dataset")

# Add nodes
node1 = graph.add_node(plan.id, name="Load Data", action="load_data")
node2 = graph.add_node(plan.id, name="Transform", action="transform")
node3 = graph.add_node(plan.id, name="Analyze", action="analyze")

# Connect nodes
graph.add_edge(plan.id, f"{plan.id}_start", node1.id)
graph.add_edge(plan.id, node1.id, node2.id)
graph.add_edge(plan.id, node2.id, node3.id)
graph.add_edge(plan.id, node3.id, f"{plan.id}_end")

# Execute
result = await graph.execute_plan(plan.id)
```

### 2. Cognitive Memory

FAISS-powered semantic memory with human-inspired features:

- Episodic, semantic, and procedural memory types
- Working memory for current context
- Memory consolidation and forgetting
- Importance-based retention

```python
from aion.systems.memory import CognitiveMemorySystem

memory = CognitiveMemorySystem()
await memory.initialize()

# Store memories
await memory.store(
    content="The Eiffel Tower is in Paris",
    importance=0.8,
    metadata={"topic": "geography"}
)

# Search semantically
results = await memory.search(
    query="Famous landmarks in France",
    limit=10
)

# Recall and synthesize
response = await memory.recall("French architecture")
```

### 3. Tool Orchestration

Intelligent coordination of multiple tools:

- Extensible tool registry
- Rate limiting and timeout management
- Parallel execution
- Tool learning and suggestion

```python
from aion.systems.tools import ToolOrchestrator

orchestrator = ToolOrchestrator()
await orchestrator.initialize()

# Execute a single tool
result = await orchestrator.execute(
    tool_name="web_fetch",
    params={"url": "https://example.com"}
)

# Execute multiple tools in parallel
results = await orchestrator.execute_parallel([
    ("calculator", {"expression": "sqrt(16)"}),
    ("datetime", {"operation": "now"}),
])

# Create tool chains
chain = orchestrator.create_chain(
    name="Web Analysis",
    steps=[
        ("web_fetch", {"url": "$input_url"}),
        ("text_transform", {"text": "$last_result.content", "operation": "count_words"}),
    ]
)
```

### 4. Self-Improvement Engine

Autonomous evolution with safety bounds:

- Hypothesis-driven optimization
- Safe parameter adjustment
- Rollback on degradation
- Human approval gates

```python
from aion.systems.evolution import SelfImprovementEngine

engine = SelfImprovementEngine(
    safety_threshold=0.95,
    require_approval=True
)
await engine.initialize()

# Register parameters for optimization
engine.register_parameter(
    name="temperature",
    current_value=0.7,
    bounds=(0.1, 1.0)
)

# Start improvement loop
await engine.start_improvement_loop()

# Check pending approvals
approvals = engine.get_pending_approvals()
```

### 5. Visual Cortex

Multi-model computer vision with reasoning:

- Object detection (DETR)
- Image captioning (BLIP)
- Visual question answering
- Scene graph construction
- Visual memory

```python
from aion.systems.vision import VisualCortex

cortex = VisualCortex()
await cortex.initialize()

# Full analysis
result = await cortex.process(
    image_path="image.jpg",
    query="What objects are in this image?",
    store_in_memory=True
)

# Get scene description
description = await cortex.describe("image.jpg")

# Answer visual questions
answer = await cortex.answer("image.jpg", "Is there a dog in the image?")

# Compare images
comparison = await cortex.compare("image1.jpg", "image2.jpg")
```

## 🔐 Security

AION includes comprehensive security features:

- **Risk Classification**: Automatic risk assessment of operations
- **Approval Gates**: Human approval for high-risk operations
- **Audit Logging**: Complete audit trail of all actions
- **Emergency Stop**: Immediate halt mechanism
- **Checkpoints**: State preservation for rollback

```python
from aion.core.security import SecurityManager, RiskLevel

security = SecurityManager(
    require_approval_for_high_risk=True,
    auto_approve_low_risk=True
)

# Check authorization
authorized, reason = await security.authorize(
    operation="delete_file",
    description="Delete user data",
    details={"file": "/data/user.json"}
)

# Emergency stop
security.emergency_stop("Critical error detected")
```

## 🛠️ Configuration

AION uses environment-based configuration with Pydantic:

```python
from aion.core.config import AIONConfig

config = AIONConfig(
    llm={"provider": "openai", "model": "gpt-4-turbo-preview"},
    memory={"embedding_model": "all-MiniLM-L6-v2"},
    security={"require_approval_for_high_risk": True},
)
```

Or via environment variables:

```bash
export AION_LLM__PROVIDER=anthropic
export AION_LLM__MODEL=claude-3-opus-20240229
export ANTHROPIC_API_KEY=your-api-key
```

## 📖 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System information |
| `/health` | GET | Health check |
| `/status` | GET | Detailed status |
| `/process` | POST | Process user request |
| `/emergency-stop` | POST | Activate emergency stop |

### Planning Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/planning/plans` | GET | List plans |
| `/planning/plans` | POST | Create plan |
| `/planning/plans/{id}` | GET | Get plan |
| `/planning/plans/{id}/execute` | POST | Execute plan |
| `/planning/plans/{id}/visualize` | GET | Visualize plan |

### Memory Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memory/store` | POST | Store memory |
| `/memory/search` | POST | Search memories |
| `/memory/recall` | POST | Recall information |
| `/memory/consolidate` | POST | Trigger consolidation |

### Tool Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools` | GET | List tools |
| `/tools/execute` | POST | Execute tool |
| `/tools/execute-parallel` | POST | Execute multiple tools |
| `/tools/suggest` | POST | Suggest tools for task |

### Vision Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vision/analyze` | POST | Full image analysis |
| `/vision/detect` | POST | Object detection |
| `/vision/describe` | POST | Image description |
| `/vision/answer` | POST | Visual QA |
| `/vision/compare` | POST | Compare images |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_basic.py -v

# Run with coverage
pytest tests/ --cov=aion --cov-report=html
```

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| Memory search | <100ms for 1M+ memories |
| Planning | 10+ step plans in <1s |
| Tool execution | 5+ tools in parallel |
| Vision processing | <2s per image |
| API response | <200ms for simple queries |

## 🔮 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AION Kernel                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   LLM       │  │  Security   │  │   Configuration     │ │
│  │  Adapter    │  │  Manager    │  │   Management        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────────┐
│   Planning    │  │    Memory     │  │      Tools        │
│    Graph      │  │    System     │  │   Orchestrator    │
│  (NetworkX)   │  │   (FAISS)     │  │                   │
└───────────────┘  └───────────────┘  └───────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────────┐
│  Evolution    │  │    Visual     │  │     FastAPI       │
│   Engine      │  │    Cortex     │  │       API         │
│              │  │ (DETR, BLIP)  │  │                   │
└───────────────┘  └───────────────┘  └───────────────────┘
```

## 📜 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -am 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 🆘 Support

- Issues: [GitHub Issues](https://github.com/your-org/aion/issues)
- Documentation: [docs/](./docs/)
- Discord: [Join our community](#)

---

**AION** - Building the future of artificial intelligence, one cognitive system at a time. 🧠✨
