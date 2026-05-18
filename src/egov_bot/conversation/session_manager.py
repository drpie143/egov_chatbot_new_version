from __future__ import annotations

from dataclasses import dataclass, field

from egov_bot.schemas.common import Source


@dataclass
class ConversationEntry:
    role: str
    content: str
    context: str = ""
    parent_id: str | None = None
    sources: list[Source] = field(default_factory=list)


class SessionManager:
    def __init__(self, max_entries: int = 20) -> None:
        self.max_entries = max_entries
        self._sessions: dict[str, list[ConversationEntry]] = {}

    def get(self, session_id: str) -> list[ConversationEntry]:
        return self._sessions.setdefault(session_id, [])

    def clear(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        context: str,
        parent_id: str | None,
        sources: list[Source],
    ) -> None:
        history = self.get(session_id)
        history.extend(
            [
                ConversationEntry(role="user", content=question),
                ConversationEntry(
                    role="assistant",
                    content=answer,
                    context=context,
                    parent_id=parent_id,
                    sources=sources,
                ),
            ]
        )
        if len(history) > self.max_entries:
            del history[:-self.max_entries]

    def last_context(self, session_id: str) -> ConversationEntry | None:
        for entry in reversed(self.get(session_id)):
            if entry.role == "assistant" and entry.context:
                return entry
        return None

    def history_text(self, session_id: str, max_turns: int = 6) -> str:
        entries = self.get(session_id)[-max_turns * 2 :]
        lines = []
        for entry in entries:
            label = "User" if entry.role == "user" else "Assistant"
            lines.append(f"{label}: {entry.content}")
        return "\n".join(lines)

