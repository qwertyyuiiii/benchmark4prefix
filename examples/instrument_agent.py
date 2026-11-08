"""Copy these functions into your experiment runner; no SDK is bundled."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe_capture import Recorder


def openai_chat_call(client, recorder, payload):
    """OpenAI-compatible Chat Completions; payload includes full history/tools."""
    return recorder.call(
        "openai-chat", payload,
        lambda: client.chat.completions.create(**payload),
    )


def anthropic_call(client, recorder, payload):
    """Anthropic Messages; keep system and tools inside the supplied payload."""
    return recorder.call(
        "anthropic", payload,
        lambda: client.messages.create(**payload),
    )


# Example usage inside your existing runner:
#
# recorder = Recorder("logs/math-session-001.jsonl",
#                     session_id="math-session-001", agent="your-agent")
# payload = {"model": YOUR_MODEL, "messages": COMPLETE_MESSAGES,
#            "tools": COMPLETE_TOOL_DEFINITIONS, "stream": False}
# response = openai_chat_call(client, recorder, payload)
#
# When your application performs history compression:
# recorder.context_event("history_compaction", {"reason": "context_budget"})
# The following request must contain the ACTUAL compacted messages.
#
# Async SDK calls can use recorder.start_call / finish_call / fail_call around
# await client.... Do not pass an awaitable directly into recorder.call().


if __name__ == "__main__":
    print("Integration helpers only; import into your existing agent runner. No API was called.")
