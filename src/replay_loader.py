"""
Loads a Blockscope recording session and provides tick-by-tick access to:
  - Player position/rotation per tick
  - Blocks seen per tick (progressive world building)
  - Block changes per tick
"""

import json
import os
from collections import defaultdict


class ReplayLoader:
    def __init__(self, session_dir: str):
        self.session_dir = session_dir

        # Load metadata
        with open(os.path.join(session_dir, 'metadata.json')) as f:
            self.metadata = json.load(f)

        # Load ticks (player state per tick)
        self.ticks = []
        with open(os.path.join(session_dir, 'ticks.jsonl')) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.ticks.append(json.loads(line))

        # Load world events grouped by tick
        self.block_events_by_tick: dict[int, list] = defaultdict(list)
        with open(os.path.join(session_dir, 'world_events.jsonl')) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                tick = event.get('tick', 0)
                self.block_events_by_tick[tick].append(event)

        self.max_tick = max(t['tick'] for t in self.ticks) if self.ticks else 0

        # Load frame mapping (frame index -> tick number) if available
        # Each line: {"frame": N, "tick": T}
        self.frame_to_tick: list[int] = []  # index = frame, value = tick
        frame_mapping_path = os.path.join(session_dir, 'frame_mapping.jsonl')
        if os.path.exists(frame_mapping_path):
            with open(frame_mapping_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self.frame_to_tick.append(entry['tick'])
        self.has_frame_mapping = len(self.frame_to_tick) > 0

    def get_player_state(self, tick: int) -> dict | None:
        """Get player state at a given tick."""
        if 0 <= tick < len(self.ticks):
            return self.ticks[tick]
        return None

    def get_initial_player_pos(self) -> tuple[float, float, float]:
        """Get player position at tick 0."""
        if self.ticks:
            p = self.ticks[0]['player']
            return (p['x'], p['y'], p['z'])
        return (0.0, 64.0, 0.0)

    def get_events_for_tick(self, tick: int) -> list:
        """Get all world events for a given tick."""
        return self.block_events_by_tick.get(tick, [])

    def get_all_unique_block_ids(self) -> set[str]:
        """Scan all events and return unique block IDs (for pre-registering textures)."""
        block_ids = set()
        for events in self.block_events_by_tick.values():
            for event in events:
                bid = event.get('blockId', '')
                if bid and bid != 'minecraft:air':
                    block_ids.add(bid)
        return block_ids

    def get_all_unique_block_variants(self) -> set[tuple[str, str]]:
        """Scan all events and return unique (block_id, properties) pairs."""
        variants = set()
        for events in self.block_events_by_tick.values():
            for event in events:
                bid = event.get('blockId', '')
                props = event.get('blockStateProperties', '')
                if bid and bid != 'minecraft:air' and props:
                    variants.add((bid, props))
        return variants
