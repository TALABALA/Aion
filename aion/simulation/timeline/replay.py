"""AION Replay Engine - Deterministic event replay and recording.

Provides:
- ReplayEngine: Records simulation events and replays them deterministically.
- Supports selective replay (filter by type, source, time range).
- Speed control and step-through.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import structlog

from aion.simulation.types import EventType, SimulationEvent

if TYPE_CHECKING:
    from aion.simulation.world.engine import WorldEngine

logger = structlog.get_logger(__name__)


@dataclass
class ReplaySession:
    """A recorded or active replay session."""

    id: str = ""
    events: List[SimulationEvent] = field(default_factory=list)
    from_tick: int = 0
    to_tick: int = 0
    from_snapshot_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReplayEngine:
    """Deterministic event replay engine.

    Features:
    - Record events during simulation.
    - Replay recorded events from a snapshot.
    - Selective replay (filter by event type, source, time).
    - Step-by-step replay.
    - Replay speed control.
    - Replay hooks for observation.
    """

    def __init__(self, world_engine: "WorldEngine") -> None:
        self.world_engine = world_engine

        # Recording state
        self._recording = False
        self._recorded_events: List[SimulationEvent] = []
        self._record_start_tick: int = 0

        # Sessions
        self._sessions: Dict[str, ReplaySession] = {}

        # Replay hooks
        self._on_replay_event: List[Callable] = []
        self._on_replay_tick: List[Callable] = []

    # -- Recording --

    def start_recording(self, snapshot_id: Optional[str] = None) -> None:
        """Start recording events."""
        self._recording = True
        self._recorded_events.clear()
        self._record_start_tick = self.world_engine.state.tick
        logger.debug("replay_recording_started", tick=self._record_start_tick)

    def stop_recording(self) -> ReplaySession:
        """Stop recording and return the session."""
        self._recording = False
        session = ReplaySession(
            id=f"replay_{self._record_start_tick}_{self.world_engine.state.tick}",
            events=list(self._recorded_events),
            from_tick=self._record_start_tick,
            to_tick=self.world_engine.state.tick,
        )
        self._sessions[session.id] = session
        self._recorded_events.clear()
        logger.debug(
            "replay_recording_stopped",
            session_id=session.id,
            event_count=len(session.events),
        )
        return session

    def record_event(self, event: SimulationEvent) -> None:
        """Record a single event (called during simulation)."""
        if self._recording:
            self._recorded_events.append(event)

    @property
    def is_recording(self) -> bool:
        return self._recording

    # -- Replay --

    async def replay(
        self,
        session: ReplaySession,
        from_snapshot_id: Optional[str] = None,
        speed: float = 1.0,
        event_filter: Optional[Callable[[SimulationEvent], bool]] = None,
    ) -> List[SimulationEvent]:
        """Replay a recorded session.

        Args:
            session: The replay session to replay.
            from_snapshot_id: Snapshot to restore before replay.
            speed: Replay speed multiplier (>1 = faster).
            event_filter: Optional filter for selective replay.

        Returns:
            All events produced during replay.
        """
        # Restore snapshot if provided
        if from_snapshot_id:
            # Caller should handle snapshot restoration via TimelineManager
            pass

        events = session.events
        if event_filter:
            events = [e for e in events if event_filter(e)]

        # Sort by sequence number for deterministic ordering
        events.sort(key=lambda e: e.sequence_number)

        all_produced: List[SimulationEvent] = []

        # Group events by tick
        events_by_tick: Dict[int, List[SimulationEvent]] = {}
        for event in events:
            tick = event.tick
            if tick not in events_by_tick:
                events_by_tick[tick] = []
            events_by_tick[tick].append(event)

        for tick in sorted(events_by_tick.keys()):
            tick_events = events_by_tick[tick]

            for event in tick_events:
                self.world_engine.emit_event(event)

                for hook in self._on_replay_event:
                    try:
                        hook(event)
                    except Exception:
                        pass

            produced = await self.world_engine.step()
            all_produced.extend(produced)

            for hook in self._on_replay_tick:
                try:
                    hook(tick, produced)
                except Exception:
                    pass

            if speed < 100:
                await asyncio.sleep(0.001 / speed)

        return all_produced

    async def replay_step(
        self,
        session: ReplaySession,
        step_index: int,
    ) -> List[SimulationEvent]:
        """Replay a single step from a session."""
        if step_index >= len(session.events):
            return []

        event = session.events[step_index]
        self.world_engine.emit_event(event)
        return await self.world_engine.step()

    async def replay_events(
        self,
        events: List[SimulationEvent],
        from_snapshot_id: Optional[str] = None,
    ) -> List[SimulationEvent]:
        """Replay a raw list of events."""
        session = ReplaySession(events=events)
        return await self.replay(session, from_snapshot_id=from_snapshot_id)

    # -- Selective Replay --

    def filter_by_type(self, *event_types: EventType) -> Callable[[SimulationEvent], bool]:
        """Create filter for specific event types."""
        type_set = set(event_types)
        return lambda e: e.type in type_set

    def filter_by_source(self, source_id: str) -> Callable[[SimulationEvent], bool]:
        """Create filter for a specific source."""
        return lambda e: e.source_id == source_id

    def filter_by_tick_range(self, start: int, end: int) -> Callable[[SimulationEvent], bool]:
        """Create filter for a tick range."""
        return lambda e: start <= e.tick <= end

    # -- Session Management --

    def get_session(self, session_id: str) -> Optional[ReplaySession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    def delete_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    # -- Hooks --

    def on_replay_event(self, hook: Callable) -> None:
        self._on_replay_event.append(hook)

    def on_replay_tick(self, hook: Callable) -> None:
        self._on_replay_tick.append(hook)

    def clear(self) -> None:
        self._sessions.clear()
        self._recorded_events.clear()
        self._recording = False
        self._on_replay_event.clear()
        self._on_replay_tick.clear()
