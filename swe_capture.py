#!/usr/bin/env python3
"""Local, dependency-free capture and SWE-chat-core export. Python >= 3.10."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

VERSION = "1.3.0"
SCHEMA_SOURCE = "https://huggingface.co/datasets/SALT-NLP/SWE-chat/blob/main/README.md"
CONVERSATION_COLUMNS = """turn_id session_id checkpoint_pk repo_id user_id
turn_number conversation_turn_number source_entry_uuid response_id
content_block_index role turn_type is_conversational content model timestamp
input_tokens output_tokens cache_creation_input_tokens cache_read_input_tokens
is_continuation is_first_turn word_count char_count tool_name tool_call_id
file_path command pattern tool_input_json orchestration_kind category bash_category
queue_op_subtype agent strategy language prompt_intent prompt_pushback""".split()
SESSION_COLUMNS = """session_id repo_id owner_id user_id agent strategy created_at
model api_call_count turn_count prompt_count input_tokens output_tokens
cache_creation_tokens cache_read_tokens duration_seconds transcript_path
canonical_checkpoint_pk kind""".split()
ROLES = {"user", "assistant", "tool_use", "tool_result", "metadata"}
TYPES = {"user_prompt", "system_injected", "assistant_response", "assistant_thinking",
         "tool_use", "tool_result", "summary", "system_event", "file_snapshot",
         "progress", "queue_operation"}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def timestamp(value):
    """Unknown stays null. Supplied naive times are rejected instead of guessed."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a timestamp")
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, timezone.utc)
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("timestamp needs timezone: " + value)
    else:
        raise ValueError("unsupported timestamp")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def utcnow():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def jsonable(response):
    if isinstance(response, dict):
        return copy.deepcopy(response)
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    raise TypeError("Pass a complete JSON response; assemble streaming chunks first")


def read_jsonl(path):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{n}: invalid JSON") from exc
    return result


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(dumps(row) + "\n")


class Recorder:
    """Record payloads owned by the calling application, never auth headers.

    One session per file. Each append is process-locked on POSIX. Raw payloads
    are not redacted: only use content approved for your data collection.
    """
    def __init__(self, path, session_id=None, agent="local-agent", repo_id=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())
        self.agent = agent
        self.repo_id = repo_id
        self._lock = threading.RLock()
        self._pending = {}
        existing = read_jsonl(self.path) if self.path.exists() else []
        ids = {r.get("session_id") for r in existing}
        if ids and ids != {self.session_id}:
            raise ValueError("capture file belongs to a different session")

    def _append(self, row):
        row = {"capture_version": VERSION, "session_id": self.session_id,
               "agent": self.agent, "repo_id": self.repo_id, **row}
        payload = (dumps(row) + "\n").encode("utf-8")
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                try:
                    import fcntl
                except ImportError:
                    fcntl = None
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                view = memoryview(payload)
                while view:
                    n = os.write(fd, view)
                    if not n:
                        raise OSError("short capture write")
                    view = view[n:]
                os.fsync(fd)
            finally:
                os.close(fd)

    def start_call(self, provider, request, *, replay_metadata=None):
        if provider not in {"openai-chat", "anthropic"}:
            raise ValueError("provider must be openai-chat or anthropic")
        call_id = str(uuid.uuid4())
        started_at = utcnow()
        model = request.get("model")
        self._append({"event": "request_start", "call_id": call_id,
                      "timestamp": started_at, "provider": provider,
                      "model": model,
                      "request": copy.deepcopy(request),
                      "replay_metadata": replay_metadata})
        self._pending[call_id] = {"model": model, "started_at": started_at}
        return call_id

    def finish_call(self, call_id, response):
        if call_id not in self._pending:
            raise ValueError("unknown or already finished call_id")
        body = jsonable(response)
        pending = self._pending[call_id]
        self._append({"event": "request_end", "call_id": call_id,
                      "timestamp": pending["started_at"],
                      "model": pending["model"] or body.get("model"), "response": body})
        del self._pending[call_id]

    def fail_call(self, call_id, exc):
        if call_id not in self._pending:
            raise ValueError("unknown or already finished call_id")
        self._append({"event": "request_error", "call_id": call_id,
                      "timestamp": self._pending[call_id]["started_at"],
                      "model": self._pending[call_id]["model"],
                      "error_type": type(exc).__name__})
        del self._pending[call_id]

    def call(self, provider, request, invoke, *, replay_metadata=None):
        """invoke() calls your SDK. Its original response is returned unchanged."""
        if request.get("stream"):
            raise ValueError("use start_call/finish_call and assemble streaming chunks yourself")
        cid = self.start_call(provider, request, replay_metadata=replay_metadata)
        try:
            response = invoke()
            self.finish_call(cid, response)
            return response
        except Exception as exc:
            if cid in self._pending:
                self.fail_call(cid, exc)
            raise

    def context_event(self, kind, details):
        self._append({"event": "context_event", "timestamp": utcnow(),
                      "kind": kind, "details": copy.deepcopy(details)})


def usage_fields(usage):
    usage = usage or {}
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens",
             (usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
    }


def safe_name(session_id):
    if re.fullmatch(r"[A-Za-z0-9_-]{1,120}", session_id):
        return session_id
    return "session-" + hashlib.sha256(session_id.encode()).hexdigest()[:24]


def history_key(message):
    # Only a conversation-display de-dup key; never a token/KV hash.
    keys = {"role", "content", "name", "tool_calls", "tool_call_id", "function_call"}
    return {k: v for k, v in message.items() if k in keys and v is not None}


def message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(p.get("type") == "text" for p in content):
        return "\n".join(p.get("text", "") for p in content)
    return dumps(content)


def semantic_message(message, index, metadata):
    """Apply explicit framework annotations to analysis only, never raw payloads."""
    matches = [a for a in (metadata or {}).get("message_annotations", []) if a.get("index") == index]
    if not matches:
        return message
    if len(matches) != 1:
        raise ValueError("duplicate framework message annotation")
    annotation = matches[0]
    digest = hashlib.sha256(message_text(message).encode("utf-8")).hexdigest()
    if annotation.get("content_sha256") != digest:
        raise ValueError("framework annotation content hash mismatch")
    result = copy.deepcopy(message)
    if annotation.get("kind") == "tool_result":
        result.update(role="tool", tool_call_id=annotation.get("tool_call_id"),
                      name=annotation.get("tool_name"))
    elif annotation.get("kind") == "system_injected":
        result["is_human"] = False
    elif annotation.get("kind") == "history_rendering":
        result["_framework_history"] = True
    else:
        raise ValueError("unsupported framework annotation kind")
    result["_semantic_kind"] = annotation.get("kind")
    return result


class Session:
    def __init__(self, session_id, *, agent=None, repo_id=None, provenance=None):
        self.session_id = str(session_id)
        self.agent = agent
        self.repo_id = repo_id
        self.provenance = provenance or {}
        self.rows = []
        self.raw = []
        self.annotations = []
        self.calls = []
        self.tools = {}
        self.conv_index = 0
        self.api_calls = None
        self.history_changes = 0
        self._used_response_ids = set()

    def add(self, role, turn_type, content, **fields):
        if role not in ROLES or turn_type not in TYPES:
            raise ValueError(f"unsupported SWE-chat row: {role}/{turn_type}")
        if not isinstance(content, str):
            content = dumps(content)
        original = content
        conversational = turn_type in {"user_prompt", "assistant_response"}
        row = dict.fromkeys(CONVERSATION_COLUMNS)
        row.update(turn_id=f"{self.session_id}#{len(self.rows)}", session_id=self.session_id,
                   repo_id=self.repo_id, turn_number=len(self.rows),
                   conversation_turn_number=self.conv_index if conversational else None,
                   role=role, turn_type=turn_type, is_conversational=conversational,
                   content=content, content_block_index=0, agent=self.agent,
                   is_continuation=turn_type == "user_prompt" and content.startswith("This session is being continued"),
                   is_first_turn=conversational and self.conv_index == 0,
                   word_count=len(content.split()), char_count=len(content))
        row.update(fields)
        row["timestamp"] = timestamp(row["timestamp"])
        self.rows.append(row)
        self.annotations.append({"turn_id": row["turn_id"],
                                 "content_truncated": content != original,
                                 "original_utf8_bytes": len(original.encode("utf-8"))})
        if conversational:
            self.conv_index += 1
        return row

    def message(self, message):
        """Normalize supplied OpenAI messages or Anthropic content blocks."""
        start = len(self.rows)
        role = message.get("role")
        content = message.get("content")
        common = {"timestamp": message.get("timestamp"),
                  "source_entry_uuid": message.get("id"),
                  "response_id": message.get("response_id"),
                  "model": message.get("model")}
        if message.get("_framework_history"):
            self.add("metadata", "system_event", message_text(message), **common)
        elif role in {"system", "developer"}:
            self.add("metadata", "system_event", content if content is not None else "", **common)
        elif role in {"tool", "function"}:
            cid = message.get("tool_call_id")
            self.add("tool_result", "tool_result", content if content is not None else "",
                     tool_call_id=cid, tool_name=message.get("name") or self.tools.get(cid), **common)
        elif role in {"user", "assistant"}:
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
            if not isinstance(parts, list):
                raise ValueError("content must be text, blocks, or null")
            for i, part in enumerate(parts):
                fields = {**common, "content_block_index": i}
                if isinstance(part, str):
                    part = {"type": "text", "text": part}
                typ = part.get("type")
                if typ in {"text", "input_text", "output_text"}:
                    text = part.get("text", "")
                    if not isinstance(text, str):
                        raise ValueError("text block is not a string")
                    turn_type = "assistant_response" if role == "assistant" else (
                        "system_injected" if message.get("is_human") is False else "user_prompt")
                    self.add(role, turn_type, text, **fields)
                elif typ == "tool_use":
                    cid = part.get("id")
                    self.tools[cid] = part.get("name")
                    self._tool(part.get("name"), cid, part.get("input", {}), fields)
                elif typ == "tool_result":
                    cid = part.get("tool_use_id")
                    result = part.get("content", "")
                    if isinstance(result, list) and all(p.get("type") == "text" for p in result):
                        result = "\n".join(p.get("text", "") for p in result)
                    self.add("tool_result", "tool_result", result,
                             tool_call_id=cid, tool_name=self.tools.get(cid), **fields)
                elif typ in {"thinking", "reasoning"}:
                    # Only explicitly supplied local logs are parsed. Never inferred.
                    self.add("assistant", "assistant_thinking",
                             part.get("thinking", part.get("text", "")), **fields)
                else:
                    self.add("metadata", "system_event", {"non_text_content": part}, **fields)
            for i, call in enumerate(message.get("tool_calls") or []):
                fn = call.get("function") or {}
                cid = call.get("id")
                self.tools[cid] = fn.get("name")
                self._tool(fn.get("name"), cid, fn.get("arguments", "{}"),
                           {**common, "content_block_index": len(parts) + i})
        else:
            raise ValueError(f"unsupported input role {role!r}")
        rows = self.rows[start:]
        for annotation in self.annotations[start:]:
            annotation.update(
                call_id=message.get("_call_id"),
                timestamp_source=message.get("_timestamp_source") or (
                    "source_message" if message.get("timestamp") is not None else "unavailable"),
                semantic_kind=message.get("_semantic_kind"),
            )
        usage = message.get("usage")
        response_id = message.get("response_id")
        if usage and (not response_id or response_id not in self._used_response_ids):
            target = next((r for r in rows if r["turn_type"] == "assistant_response"), None)
            target = target or next((r for r in rows if r["turn_type"] == "tool_use"), None)
            if target is not None:
                target.update(usage_fields(usage))
                if response_id:
                    self._used_response_ids.add(response_id)

    def _tool(self, name, cid, args, fields):
        if isinstance(args, str):
            raw_args = args
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        else:
            raw_args = dumps(args)
        args = args if isinstance(args, dict) else {}
        self.add("tool_use", "tool_use", raw_args, tool_name=name, tool_call_id=cid,
                 tool_input_json=raw_args, file_path=args.get("file_path", args.get("path")),
                 command=args.get("command", args.get("cmd")), pattern=args.get("pattern"),
                 **fields)


def load_messages(path, override_id=None):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    messages = obj if isinstance(obj, list) else obj.get("messages")
    if not isinstance(messages, list):
        raise ValueError("expected a message list or {messages: [...]}")
    meta = obj if isinstance(obj, dict) else {}
    sid = override_id or meta.get("session_id") or str(uuid.uuid4())
    session = Session(sid, agent=meta.get("agent"), repo_id=meta.get("repo_id"),
                      provenance={"format": "messages", **meta.get("provenance", {}),
                                  "actual_request_payloads": False})
    for msg in messages:
        session.raw.append({"event": "source_message", "message": msg})
        session.message(msg)
    return [session]


def load_chatgpt(path, conversation_id=None):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    chats = obj if isinstance(obj, list) else [obj]
    selected = [c for c in chats if not conversation_id or c.get("id", c.get("conversation_id")) == conversation_id]
    if not selected:
        raise ValueError("conversation_id not found")
    sessions = []
    for chat in selected:
        mapping = chat.get("mapping")
        node_id = chat.get("current_node")
        if not isinstance(mapping, dict) or node_id not in mapping:
            raise ValueError("ChatGPT export needs mapping and a valid current_node")
        chain, visited = [], set()
        while node_id is not None:
            if node_id in visited or node_id not in mapping:
                raise ValueError("cycle or missing parent in ChatGPT mapping")
            visited.add(node_id)
            node = mapping[node_id]
            chain.append(node)
            node_id = node.get("parent")
        sid = chat.get("id", chat.get("conversation_id")) or str(uuid.uuid4())
        session = Session(sid, agent="ChatGPT", provenance={
            "format": "chatgpt_export_active_branch", "actual_request_payloads": False,
            "excluded_branches": len(mapping) - len(chain), "excluded_nonpublic_entries": 0})
        for node in reversed(chain):
            msg = node.get("message")
            if not msg:
                continue
            channel = msg.get("channel") or (msg.get("metadata") or {}).get("channel")
            c = msg.get("content") or {}
            if channel in {"analysis", "reasoning", "justify", "confidence"} or c.get("content_type") in {
                    "thoughts", "reasoning_recap", "model_editable_context"}:
                session.provenance["excluded_nonpublic_entries"] += 1
                continue
            author = msg.get("author") or {}
            role = author.get("role")
            if role not in {"user", "assistant", "tool"}:
                session.provenance["excluded_nonpublic_entries"] += 1
                continue
            parts = c.get("parts")
            if parts is None:
                text = c.get("text")
                parts = [text] if isinstance(text, str) else [dumps(c)]
            content = [{"type": "text", "text": p} if isinstance(p, str)
                       else {"type": "source_attachment", "data": p} for p in parts]
            if role == "tool":
                content = "\n".join(p["text"] if p["type"] == "text" else dumps(p) for p in content)
            public = {"role": role, "content": content, "id": msg.get("id"),
                      "timestamp": msg.get("create_time"),
                      "model": (msg.get("metadata") or {}).get("model_slug"),
                      "name": author.get("name")}
            session.raw.append({"event": "export_message", "message": public})
            session.message(public)
        sessions.append(session)
    return sessions


def load_claude(path, override_id=None):
    records = read_jsonl(path)
    ids = {r.get("sessionId") for r in records if r.get("sessionId")}
    if len(ids) > 1:
        raise ValueError("use one native session per file")
    sid = override_id or (next(iter(ids)) if ids else str(uuid.uuid4()))
    session = Session(sid, agent="Claude Code", provenance={
        "format": "claude_code_native_jsonl", "actual_request_payloads": False})
    for record in records:
        session.raw.append(record)
        typ = record.get("type")
        if typ in {"user", "assistant"} and isinstance(record.get("message"), dict):
            msg = copy.deepcopy(record["message"])
            msg.setdefault("role", typ)
            msg["timestamp"] = record.get("timestamp")
            msg["response_id"] = msg.get("id") if typ == "assistant" else None
            msg["id"] = record.get("uuid")
            session.message(msg)
        else:
            turn_type = {"summary": "summary", "progress": "progress",
                         "file-history-snapshot": "file_snapshot",
                         "queue-operation": "queue_operation"}.get(typ, "system_event")
            session.add("metadata", turn_type, record, timestamp=record.get("timestamp"),
                        source_entry_uuid=record.get("uuid"))
    return [session]


def response_message(provider, response, ts, model=None):
    if provider == "openai-chat":
        choices = response.get("choices") or []
        if len(choices) != 1:
            raise ValueError("openai-chat export requires exactly one completion choice")
        msg = copy.deepcopy(choices[0]["message"])
    elif provider == "anthropic":
        msg = {"role": "assistant", "content": response.get("content", [])}
    else:
        raise ValueError("unsupported provider")
    msg.update(timestamp=ts, response_id=response.get("id"),
               model=model or response.get("model"), usage=response.get("usage"),
               _timestamp_source="call_started")
    return msg


def load_api(path, override_id=None):
    records = read_jsonl(path)
    ids = {r.get("session_id") for r in records}
    if len(ids) != 1 or None in ids:
        raise ValueError("capture file must contain exactly one known session_id")
    first = records[0]
    session = Session(override_id or first["session_id"], agent=first.get("agent"),
                      repo_id=first.get("repo_id"), provenance={
        "format": "actual_api_payloads", "actual_request_payloads": True,
        "analysis_projection": "request-start order; original chronology stays in raw transcript",
        "token_exact_replay_ready": False,
        "note": "Raw payloads preserved; tokenizer/template/version still required for exact replay."})
    session.raw = records
    starts, ends = {}, {}
    pending, overlapping = set(), 0
    for record in records:
        event = record.get("event")
        if event == "request_start":
            cid = record["call_id"]
            if cid in starts:
                raise ValueError("duplicate request_start call_id")
            if pending:
                overlapping += 1
            pending.add(cid)
            starts[cid] = record
        elif event in {"request_end", "request_error"}:
            cid = record["call_id"]
            if cid in ends:
                raise ValueError("duplicate terminal call_id")
            pending.discard(cid)
            ends[cid] = record
    if set(ends) - set(starts):
        raise ValueError("terminal response without request_start")
    session.api_calls = len(starts)
    session.provenance["overlapping_calls"] = overlapping
    previous = []
    for record in records:
        if record.get("event") == "context_event":
            session.add("metadata", "system_event", record, timestamp=record.get("timestamp"))
            continue
        if record.get("event") != "request_start":
            continue
        begin = record
        cid = begin["call_id"]
        request = begin["request"]
        terminal = ends.get(cid)
        model = request.get("model") or begin.get("model") or (
            (terminal.get("response") or {}).get("model")
            if terminal and terminal["event"] == "request_end" else None)
        session.calls.append({
            "session_id": session.session_id, "call_id": cid,
            "provider": begin["provider"],
            "started_at": timestamp(begin.get("timestamp")),
            "model": model,
            "status": "completed" if terminal and terminal["event"] == "request_end" else (
                "error" if terminal else "incomplete"),
        })
        messages = copy.deepcopy(request.get("messages") or [])
        if begin["provider"] == "anthropic" and request.get("system") is not None:
            messages.insert(0, {"role": "system", "content": request["system"]})
        overlap = 0
        while overlap < min(len(previous), len(messages)) and history_key(previous[overlap]) == history_key(messages[overlap]):
            overlap += 1
        if overlap < len(previous):
            session.history_changes += 1
            session.add("metadata", "summary", {"event": "request_history_changed",
                        "call_id": cid, "common_message_count": overlap,
                        "note": "Observed structural change; compression is not inferred."},
                        timestamp=begin.get("timestamp"))
        for index in range(overlap, len(messages)):
            msg = semantic_message(messages[index], index, begin.get("replay_metadata"))
            msg = {**msg, "timestamp": begin.get("timestamp"),
                   "model": model, "_call_id": cid,
                   "_timestamp_source": "call_started"}
            session.message(msg)
        if terminal and terminal["event"] == "request_end":
            msg = response_message(begin["provider"], terminal["response"],
                                   begin.get("timestamp"), model)
            msg["_call_id"] = cid
            session.message(msg)
            expected = {k: v for k, v in msg.items() if k not in {
                "timestamp", "response_id", "model", "usage"} and v is not None}
            previous = messages + [expected]
        else:
            session.add("metadata", "system_event", {"call_id": cid,
                        "event": "request_error" if terminal else "missing_response",
                        "error_type": terminal.get("error_type") if terminal else None},
                        timestamp=begin.get("timestamp"), model=model)
            previous = messages
    return [session]


def export_sessions(sessions, output_dir):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty (avoid accidental overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    all_rows, session_rows, logs, annotations, quality, calls = [], [], [], [], [], []
    seen_ids = set()
    for s in sessions:
        if s.session_id in seen_ids:
            raise ValueError("duplicate session_id")
        seen_ids.add(s.session_id)
        relative = f"transcripts/{safe_name(s.session_id)}.jsonl"
        write_jsonl(out / relative, s.raw)
        rows = s.rows
        known_times = sorted((r["timestamp"] for r in rows if r["timestamp"]),
                             key=lambda t: datetime.fromisoformat(t.replace("Z", "+00:00")))
        usage_totals = {}
        for dest, origin in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                             ("cache_creation_tokens", "cache_creation_input_tokens"),
                             ("cache_read_tokens", "cache_read_input_tokens")):
            values = [r[origin] for r in rows if r[origin] is not None]
            usage_totals[dest] = sum(values) if values else None
        item = dict.fromkeys(SESSION_COLUMNS)
        item.update(session_id=s.session_id, agent=s.agent, repo_id=s.repo_id,
                    owner_id=s.repo_id.split("/")[0] if s.repo_id else None,
                    created_at=known_times[0] if known_times else None,
                    model=next((r["model"] for r in rows if r["model"]), None),
                    api_call_count=s.api_calls, turn_count=sum(r["is_conversational"] for r in rows),
                    prompt_count=sum(r["turn_type"] == "user_prompt" and not r["is_continuation"] for r in rows),
                    transcript_path=relative, **usage_totals)
        if known_times:
            item["duration_seconds"] = (datetime.fromisoformat(known_times[-1].replace("Z", "+00:00")) - datetime.fromisoformat(known_times[0].replace("Z", "+00:00"))).total_seconds()
        all_rows.extend(rows)
        session_rows.append(item)
        logs.append({"session_id": s.session_id, "transcript_path": relative,
                     "compact_transcript_path": None})
        annotations.extend(s.annotations)
        calls.extend(s.calls)
        quality.append({"session_id": s.session_id, "provenance": s.provenance,
                        "rows": len(rows), "missing_timestamp_rows": sum(r["timestamp"] is None for r in rows),
                        "missing_assistant_model_rows": sum(r["role"] == "assistant" and r["model"] is None for r in rows),
                        "observed_history_changes": s.history_changes,
                        "truncated_analysis_rows": sum(a["content_truncated"] for a in s.annotations),
                        "source_payloads_sha256": hashlib.sha256((out / relative).read_bytes()).hexdigest()})
    for name, data in (("conversations", all_rows), ("sessions", session_rows),
                       ("session_logs", logs), ("capture_annotations", annotations), ("calls", calls)):
        write_jsonl(out / f"{name}.jsonl", data)
    (out / "capture_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "schema_manifest.json").write_text(json.dumps({
        "exporter_version": VERSION, "schema": "SWE-chat-inspired full-text JSONL",
        "official_schema_source": SCHEMA_SOURCE,
        "compatibility": "Core column names retained; full text, call start timestamp, and one model field.",
        "conversations_columns": CONVERSATION_COLUMNS, "sessions_columns": SESSION_COLUMNS,
        "extensions": ["calls.jsonl", "capture_annotations.jsonl", "capture_quality.json"],
        "content_policy": "full_text_no_truncation",
        "tool_result_analysis_limit_utf8_bytes": None,
        "unknown_values": "null; no inferred timestamps/models/token counts",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return validate(out)


def validate(output_dir):
    out = Path(output_dir)
    rows = read_jsonl(out / "conversations.jsonl")
    sessions = read_jsonl(out / "sessions.jsonl")
    logs = read_jsonl(out / "session_logs.jsonl")
    ids = [s["session_id"] for s in sessions]
    errors, warnings = [], []
    if len(ids) != len(set(ids)):
        errors.append("duplicate session_id")
    turn_ids = set()
    grouped = {sid: [] for sid in ids}
    for row in rows:
        if set(row) != set(CONVERSATION_COLUMNS):
            errors.append("unexpected/missing conversation columns")
        if row.get("session_id") not in grouped:
            errors.append("orphan session reference")
            continue
        grouped[row["session_id"]].append(row)
        if row["turn_id"] in turn_ids:
            errors.append("duplicate turn_id")
        turn_ids.add(row["turn_id"])
        if row["role"] not in ROLES or row["turn_type"] not in TYPES:
            errors.append("invalid role/turn_type")
        if row["is_conversational"] != (row["turn_type"] in {"user_prompt", "assistant_response"}):
            errors.append("invalid conversational flag")
        if row["char_count"] != len(row["content"]):
            errors.append("incorrect char_count")
        try:
            timestamp(row["timestamp"])
        except (ValueError, TypeError):
            errors.append("invalid timestamp")
    for sid, data in grouped.items():
        if [r["turn_number"] for r in data] != list(range(len(data))):
            errors.append("non-contiguous turn_number")
        conv = [r for r in data if r["is_conversational"]]
        if [r["conversation_turn_number"] for r in conv] != list(range(len(conv))):
            errors.append("non-contiguous conversation_turn_number")
        tool_ids = {r["tool_call_id"] for r in data if r["turn_type"] == "tool_use" and r["tool_call_id"]}
        for r in data:
            if r["turn_type"] == "tool_result" and r["tool_call_id"] not in tool_ids:
                warnings.append(f"{sid}: tool result has no captured matching call")
    if sorted(l["session_id"] for l in logs) != sorted(ids):
        errors.append("session_logs foreign keys mismatch")
    for log in logs:
        p = (out / log["transcript_path"]).resolve()
        if not p.is_relative_to(out.resolve()) or not p.is_file():
            errors.append("invalid transcript path")
    quality = json.loads((out / "capture_quality.json").read_text(encoding="utf-8"))
    calls_path = out / "calls.jsonl"
    if calls_path.exists():
        call_ids = set()
        for call in read_jsonl(calls_path):
            if call.get("session_id") not in grouped:
                errors.append("orphan call session reference")
            if not call.get("call_id") or call["call_id"] in call_ids:
                errors.append("missing/duplicate call_id")
            call_ids.add(call.get("call_id"))
            try:
                timestamp(call.get("started_at"))
            except (ValueError, TypeError):
                errors.append("invalid call timestamp")
            if not call.get("started_at"):
                warnings.append("a call has no recorded start timestamp")
            if not call.get("model"):
                warnings.append("a call has no known model name")
    for q in quality:
        log = next((l for l in logs if l["session_id"] == q["session_id"]), None)
        if log:
            raw_path = (out / log["transcript_path"]).resolve()
            if raw_path.is_relative_to(out.resolve()) and raw_path.is_file():
                actual_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                if actual_hash != q["source_payloads_sha256"]:
                    errors.append("raw transcript checksum mismatch")
        if q["missing_timestamp_rows"]:
            warnings.append(f"{q['session_id']}: timestamps missing; not a measured arrival trace")
        if not q["provenance"].get("actual_request_payloads"):
            warnings.append(f"{q['session_id']}: dialogue only; exact API request/prefix not captured")
        if q["provenance"].get("overlapping_calls"):
            warnings.append(f"{q['session_id']}: overlapping calls; use raw per-call snapshots, not one inferred linear history")
    return {"valid": not errors, "session_count": len(sessions), "row_count": len(rows),
            "user_prompts": sum(r["turn_type"] == "user_prompt" for r in rows),
            "assistant_responses": sum(r["turn_type"] == "assistant_response" for r in rows),
            "tool_calls": sum(r["turn_type"] == "tool_use" for r in rows),
            "tool_results": sum(r["turn_type"] == "tool_result" for r in rows),
            "errors": sorted(set(errors)), "warnings": sorted(set(warnings))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="convert local logs; never crawls accounts")
    imp.add_argument("--format", required=True, choices=["messages", "chatgpt", "claude", "api"])
    imp.add_argument("--input", required=True)
    imp.add_argument("--output-dir", required=True)
    imp.add_argument("--session-id")
    imp.add_argument("--conversation-id", help="ChatGPT export active branch selector")
    check = sub.add_parser("validate")
    check.add_argument("--dataset", required=True)
    check.add_argument("--report")
    args = parser.parse_args()
    try:
        if args.command == "import":
            loaders = {"messages": load_messages, "claude": load_claude, "api": load_api}
            sessions = (load_chatgpt(args.input, args.conversation_id) if args.format == "chatgpt"
                        else loaders[args.format](args.input, args.session_id))
            result = export_sessions(sessions, args.output_dir)
        else:
            result = validate(args.dataset)
            if args.report:
                Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["valid"] else 1
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
