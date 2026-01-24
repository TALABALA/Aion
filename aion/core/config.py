"""
AION Configuration Management System

Centralized configuration for all AION subsystems with:
- Environment-based configuration
- Type-safe settings with Pydantic
- Runtime configuration updates
- Hierarchical configuration merging
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Literal
from enum import Enum
import json

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """Logging levels for AION."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: Literal["openai", "anthropic", "local", "mock"] = "openai"
    model: str = "gpt-4-turbo-preview"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 60.0
    max_retries: int = 3


class MemoryConfig(BaseModel):
    """Configuration for the Vector Memory System."""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    index_type: Literal["flat", "ivf", "hnsw"] = "flat"
    nlist: int = 100  # For IVF index
    nprobe: int = 10  # For IVF search
    ef_construction: int = 200  # For HNSW
    ef_search: int = 50  # For HNSW
    max_memories: int = 1_000_000
    consolidation_interval: int = 3600  # seconds
    importance_threshold: float = 0.3
    forgetting_rate: float = 0.01
    persistence_path: Optional[Path] = None


class PlanningConfig(BaseModel):
    """Configuration for the Planning Graph System."""
    max_plan_depth: int = 20
    max_parallel_branches: int = 5
    default_timeout: float = 300.0  # seconds per step
    checkpoint_interval: int = 1  # checkpoint every N steps
    max_retries_per_step: int = 3
    enable_visualization: bool = True
    cache_plans: bool = True


class ToolConfig(BaseModel):
    """Configuration for the Tool Orchestration System."""
    max_parallel_tools: int = 10
    default_rate_limit: float = 10.0  # requests per second
    tool_timeout: float = 60.0
    enable_learning: bool = True
    performance_tracking: bool = True
    sandbox_mode: bool = True


class EvolutionConfig(BaseModel):
    """Configuration for the Self-Improvement Engine."""
    enable_self_improvement: bool = True
    improvement_interval: int = 3600  # seconds
    min_samples_for_optimization: int = 100
    safety_threshold: float = 0.95  # minimum performance to maintain
    max_parameter_change: float = 0.1  # max % change per iteration
    hypothesis_batch_size: int = 5
    rollback_on_degradation: bool = True
    require_approval_for_changes: bool = True


class VisionConfig(BaseModel):
    """Configuration for the Visual Cortex System."""
    detection_model: str = "facebook/detr-resnet-50"
    captioning_model: str = "Salesforce/blip-image-captioning-base"
    segmentation_model: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    max_image_size: int = 1024
    enable_visual_memory: bool = True
    attention_threshold: float = 0.5
    scene_graph_enabled: bool = True


class AudioConfig(BaseModel):
    """Configuration for the Auditory Cortex System."""
    enabled: bool = True
    device: str = "auto"  # "cpu", "cuda", "mps", "auto"

    # ASR (Speech Recognition)
    whisper_model: str = "openai/whisper-large-v3"
    whisper_language: Optional[str] = None  # Auto-detect if None

    # TTS (Text-to-Speech)
    tts_model: str = "suno/bark-small"
    enable_tts: bool = True

    # Speaker Diarization
    enable_diarization: bool = True
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # Audio Event Detection
    event_detection_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    event_threshold: float = 0.3

    # Speaker Recognition
    speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"

    # Audio-Text Embeddings
    clap_model: str = "laion/larger_clap_general"

    # Memory
    enable_memory: bool = True
    memory_embedding_dim: int = 512
    memory_max_entries: int = 50000
    memory_index_path: Optional[str] = None

    # Music Analysis
    enable_music_analysis: bool = True

    # Processing limits
    max_audio_duration: float = 600.0  # 10 minutes max
    target_sample_rate: int = 16000


class SecurityConfig(BaseModel):
    """Configuration for the Security System."""
    require_approval_for_high_risk: bool = True
    auto_approve_low_risk: bool = True
    approval_timeout: float = 300.0  # seconds
    max_pending_approvals: int = 10
    audit_all_actions: bool = True
    rate_limit_requests: bool = True
    requests_per_minute: int = 60
    enable_sandboxing: bool = True
    blocked_operations: list[str] = Field(default_factory=list)


class MonitoringConfig(BaseModel):
    """Configuration for system monitoring."""
    enable_metrics: bool = True
    metrics_interval: int = 60  # seconds
    enable_tracing: bool = True
    log_level: LogLevel = LogLevel.INFO
    log_format: Literal["json", "text"] = "json"
    performance_alerts: bool = True
    alert_threshold_latency_ms: float = 1000.0
    alert_threshold_error_rate: float = 0.05


class MCPConfig(BaseModel):
    """Configuration for the MCP Integration Layer."""
    enabled: bool = True
    config_path: Path = Path("./config/mcp_servers.json")
    credentials_path: Path = Path("./config/mcp_credentials.json")
    auto_reconnect: bool = True
    health_check_interval: float = 30.0
    default_timeout: float = 30.0
    max_reconnect_attempts: int = 3

    # Server mode settings (when AION acts as MCP server)
    serve_as_mcp_server: bool = False
    mcp_server_name: str = "aion"
    mcp_server_version: str = "1.0.0"


class KnowledgeGraphConfig(BaseModel):
    """Configuration for the Knowledge Graph System."""
    # Storage settings
    store_type: Literal["sqlite", "memory"] = "sqlite"
    database_path: Optional[Path] = None  # Defaults to data_dir/knowledge.db

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Query settings
    default_query_limit: int = 100
    max_query_limit: int = 1000
    query_cache_size: int = 1000
    query_cache_ttl: int = 300  # seconds

    # Inference settings
    enable_inference: bool = True
    inference_depth: int = 3
    confidence_threshold: float = 0.5
    max_inference_iterations: int = 100

    # Hybrid search settings
    vector_weight: float = 0.4
    graph_weight: float = 0.3
    text_weight: float = 0.3
    diversity_weight: float = 0.3
    use_reranking: bool = True

    # Extraction settings
    enable_extraction: bool = True
    extraction_min_confidence: float = 0.5
    extraction_max_entities: int = 50
    extraction_max_relationships: int = 100

    # Graph analysis settings
    enable_centrality: bool = True
    enable_community_detection: bool = True
    pagerank_damping: float = 0.85

    # Temporal settings
    enable_temporal: bool = True
    default_temporal_scope: bool = False  # If True, only return currently valid facts

    # Schema settings
    schema_validation: bool = True
    schema_path: Optional[Path] = None


class ProcessConfig(BaseModel):
    """Configuration for the Process & Agent Manager System."""
    # Supervisor settings
    max_processes: int = 100
    health_check_interval: float = 5.0
    default_restart_delay: float = 1.0
    default_max_restarts: int = 5
    zombie_timeout_seconds: float = 300.0
    enable_resource_monitoring: bool = True

    # Scheduler settings
    scheduler_check_interval: float = 1.0
    max_concurrent_scheduled_tasks: int = 10

    # Worker pool settings
    worker_pool_min_workers: int = 2
    worker_pool_max_workers: int = 10
    worker_pool_max_queue_size: int = 1000
    worker_pool_enable_auto_scaling: bool = True

    # Event bus settings
    event_bus_max_history: int = 10000
    event_bus_max_dead_letters: int = 1000
    event_bus_default_ttl_seconds: Optional[int] = None

    # Persistence settings
    persistence_enabled: bool = True
    persistence_type: Literal["sqlite", "memory", "postgres"] = "sqlite"
    persistence_path: Optional[Path] = None
    persistence_cleanup_interval: int = 3600

    # Resource limits (system-wide defaults)
    default_max_memory_mb: Optional[int] = None
    default_max_tokens_per_minute: Optional[int] = 10000
    default_max_tokens_total: Optional[int] = None
    default_max_runtime_seconds: Optional[int] = None

    # System agents
    enable_health_monitor: bool = True
    enable_garbage_collector: bool = True
    enable_metrics_collector: bool = True
    enable_watchdog: bool = True

    # Health monitor settings
    health_monitor_interval: int = 30
    health_monitor_memory_threshold: float = 80.0
    health_monitor_cpu_threshold: float = 90.0

    # Garbage collector settings
    gc_interval: int = 300
    gc_max_completed_age: int = 3600

    # Metrics collector settings
    metrics_collect_interval: int = 60

    # Watchdog settings
    watchdog_check_interval: int = 30
    watchdog_heartbeat_timeout: int = 120
    watchdog_idle_timeout: int = 600


class AIONConfig(BaseSettings):
    """
    Main AION Configuration

    Loads configuration from environment variables and/or config files.
    Environment variables are prefixed with AION_ (e.g., AION_LOG_LEVEL=DEBUG)
    """

    # System identification
    instance_id: str = Field(default="aion-primary")
    environment: Literal["development", "staging", "production"] = "development"

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    debug: bool = False

    # Subsystem configurations
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)

    # Data paths
    data_dir: Path = Field(default=Path("./data"))
    checkpoints_dir: Path = Field(default=Path("./checkpoints"))
    logs_dir: Path = Field(default=Path("./logs"))

    model_config = {
        "env_prefix": "AION_",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
    }

    @field_validator("data_dir", "checkpoints_dir", "logs_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: Any) -> Path:
        """Ensure value is converted to Path."""
        if isinstance(v, str):
            return Path(v)
        return v

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [self.data_dir, self.checkpoints_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file(cls, config_path: Path) -> "AIONConfig":
        """Load configuration from a JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_data = json.load(f)

        return cls(**config_data)

    def to_file(self, config_path: Path) -> None:
        """Save configuration to a JSON file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    def get_llm_api_key(self) -> Optional[str]:
        """Get the LLM API key from config or environment."""
        if self.llm.api_key:
            return self.llm.api_key

        # Try environment variables
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }

        env_var = env_keys.get(self.llm.provider)
        if env_var:
            return os.environ.get(env_var)

        return None


# Global configuration instance (lazy loaded)
_config: Optional[AIONConfig] = None


def get_config() -> AIONConfig:
    """Get the global AION configuration instance."""
    global _config
    if _config is None:
        _config = AIONConfig()
    return _config


def set_config(config: AIONConfig) -> None:
    """Set the global AION configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to default."""
    global _config
    _config = None
