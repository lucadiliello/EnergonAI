from dataclasses import dataclass
from typing import Any, Hashable, Tuple


@dataclass
class TaskEntry:
    uids: Tuple[Hashable, ...]
    batch: Any
