"""
Manage learned dynamic handlers with clear separation of concerns.

The module now organizes storage, embedding, text analysis, scoring, and public
APIs into focused classes so each responsibility is easy to reason about and
extend.
"""

import copy
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

LOGGER = logging.getLogger(__name__)

DYNAMIC_HANDLERS_FILE = "dynamic_handlers.json"
_DYNAMIC_HANDLERS_PATH = Path(DYNAMIC_HANDLERS_FILE)

_STATUS_APPROVED = "approved"
_STATUS_PENDING = "pending"
_STATUS_DISABLED = "disabled"
_DEFAULT_STATUS = _STATUS_PENDING

_FALLBACK_MATCH_THRESHOLD = 0.52
_MAX_EMBED_TEXT_LEN = 5000
_MAX_EMBED_BACKFILL_PER_RUN = 2
_PROMPT_EMBED_CACHE_LIMIT = 32

_MONTH_PATTERN = re.compile(
    r"\b("
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|febuary|march|april|may|june|july|"
    r"august|september|october|november|december"
    r")\b"
)

_STOP_WORDS = {
    "the",
    "and",
    "what",
    "how",
    "for",
    "does",
    "can",
    "with",
    "from",
    "are",
    "was",
    "were",
    "you",
    "your",
    "will",
    "should",
    "would",
    "could",
    "tell",
    "list",
    "give",
    "show",
}


def normalize_status(status: str | None, *, default: str = _STATUS_APPROVED) -> str:
    """Normalize handler status to one of the allowed buckets."""
    value = str(status or "").strip().lower()
    if value in {_STATUS_APPROVED, _STATUS_PENDING, _STATUS_DISABLED}:
        return value
    return default


class HandlerTextProcessor:
    """Text utilities that turn prompts and handlers into comparable signals."""

    @staticmethod
    def extract_keywords(text: str | None) -> Set[str]:
        if not text or not isinstance(text, str):
            return set()
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        return {token for token in tokens if len(token) > 2 and token not in _STOP_WORDS}

    @staticmethod
    def build_embedding_text(handler_snapshot: Dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("name", "prompt", "original_prompt", "summary"):
            value = handler_snapshot.get(key)
            if value:
                parts.append(str(value))

        sql_text = handler_snapshot.get("sql")
        if sql_text:
            parts.append(f"SQL: {sql_text}")

        entities = handler_snapshot.get("entities")
        if isinstance(entities, dict) and entities:
            entity_blob = ", ".join(
                f"{key}:{value}" for key, value in entities.items() if value is not None
            )
            parts.append(f"Entities: {entity_blob}")

        keywords = handler_snapshot.get("keywords") or []
        if keywords:
            parts.append("Keywords: " + ", ".join(str(keyword) for keyword in keywords))

        blob = "\n".join(part.strip() for part in parts if part).strip()
        if len(blob) > _MAX_EMBED_TEXT_LEN:
            blob = blob[:_MAX_EMBED_TEXT_LEN]
        return blob

    @staticmethod
    def prompt_mentions_token(prompt: str, token: str | None) -> bool:
        if not prompt or not token:
            return False

        prompt_lower = prompt.lower()
        token_lower = str(token).lower()
        try:
            return re.search(rf"\b{re.escape(token_lower)}\b", prompt_lower) is not None
        except re.error:
            return False

    @staticmethod
    def prompt_requires_item_breakdown(prompt: str) -> bool:
        lower = prompt.lower()
        rank_tokens = (
            "top",
            "highest",
            "most used",
            "most-used",
            "rank",
            "ranking",
            "each item",
            "each material",
            "each raw material",
            "per item",
            "per material",
            "per raw material",
            "by item",
            "by material",
            "by raw material",
            "which raw materials",
            "what raw materials",
            "of each",
        )
        usage_tokens = ("usage", "used", "consume", "consumed", "consumption")
        return any(tok in lower for tok in rank_tokens) and any(
            tok in lower for tok in usage_tokens
        )

    @staticmethod
    def sql_supports_item_breakdown(sql_text: str | None) -> bool:
        if not sql_text or not isinstance(sql_text, str):
            return False

        sql_lower = sql_text.lower()
        has_item_columns = any(token in sql_lower for token in ("itemnmbr", "itemdesc"))
        has_grouping = "group by" in sql_lower or "partition by" in sql_lower
        has_grouping = has_grouping or "row_number" in sql_lower
        return has_item_columns and has_grouping

    @staticmethod
    def is_multi_item_usage_prompt(prompt: str) -> bool:
        lower = prompt.lower()
        raw_tokens = (
            "raw material",
            "raw materials",
            "materials",
            "ingredients",
            "components",
        )
        rank_tokens = (
            "most",
            "top",
            "highest",
            "biggest",
            "largest",
            "sum",
            "total",
            "combined",
            "aggregate",
        )
        usage_tokens = ("use", "used", "usage", "consume", "consumed", "consumption")
        has_raw = any(token in lower for token in raw_tokens)
        has_rank_or_sum = any(token in lower for token in rank_tokens)
        has_usage = any(token in lower for token in usage_tokens)
        multi_month = len(_MONTH_PATTERN.findall(lower)) >= 2
        return (has_raw and has_usage and has_rank_or_sum) or multi_month

    @staticmethod
    def handler_targets_single_item(handler_data: Dict[str, Any]) -> bool:
        entities = handler_data.get("entities") if isinstance(handler_data.get("entities"), dict) else {}
        entity_items = [value for key, value in entities.items() if key == "item" and value]
        if entity_items and len(entity_items) == 1:
            return True

        params = handler_data.get("params")
        if isinstance(params, (list, tuple)) and params:
            param0 = params[0]
            if isinstance(param0, str) and 4 <= len(param0) <= 40 and " " not in param0:
                # Avoid treating dates (YYYY-MM-DD) as items
                if re.match(r"^\d{4}-\d{2}-\d{2}$", param0):
                    return False
                return True
        return False

    @staticmethod
    def extract_handler_item_code(handler_data: Dict[str, Any]) -> str | None:
        entities = handler_data.get("entities") if isinstance(handler_data.get("entities"), dict) else {}
        entity_item = entities.get("item")
        if isinstance(entity_item, str) and entity_item.strip():
            return entity_item.strip()

        params = handler_data.get("params")
        if isinstance(params, (list, tuple)) and params:
            first = params[0]
            if isinstance(first, str):
                token = first.strip()
                if 3 <= len(token) <= 40 and " " not in token:
                    if re.match(r"^\d{4}-\d{2}-\d{2}$", token):
                        return None
                    return token
        return None


class EmbeddingService:
    """Handle embedding calls, caching, and similarity math."""

    def __init__(self, cache_limit: int = _PROMPT_EMBED_CACHE_LIMIT) -> None:
        self.cache_limit = cache_limit
        self.prompt_cache: Dict[str, List[float]] = {}

    def normalize_vector(self, vector: Any) -> List[float]:
        if not isinstance(vector, Iterable):
            return []

        normalized: list[float] = []
        for raw in vector:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isnan(value) or math.isinf(value):
                continue
            normalized.append(value)
        return normalized

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0

        length = min(len(vec_a), len(vec_b))
        if length == 0:
            return 0.0

        dot = sum(vec_a[i] * vec_b[i] for i in range(length))
        mag_a = math.sqrt(sum(vec_a[i] * vec_a[i] for i in range(length)))
        mag_b = math.sqrt(sum(vec_b[i] * vec_b[i] for i in range(length)))
        if mag_a == 0 or mag_b == 0:
            return 0.0

        try:
            return dot / (mag_a * mag_b)
        except ZeroDivisionError:
            return 0.0

    def get_prompt_embedding(self, prompt: str) -> List[float]:
        prompt_key = prompt.strip().lower()
        cached = self.prompt_cache.get(prompt_key)
        if cached:
            return cached

        vector, _ = self._generate_embedding_for_text(prompt)
        if vector:
            self._cache_prompt_embedding(prompt_key, vector)
            return vector
        return []

    def generate_handler_embedding(self, text: str) -> tuple[List[float] | None, str | None]:
        return self._generate_embedding_for_text(text)

    def _cache_prompt_embedding(self, prompt_key: str, vector: List[float]) -> None:
        self.prompt_cache[prompt_key] = vector
        over_limit = len(self.prompt_cache) - self.cache_limit
        if over_limit > 0:
            for stale_key in list(self.prompt_cache.keys())[:over_limit]:
                self.prompt_cache.pop(stale_key, None)

    def _generate_embedding_for_text(self, text: str) -> tuple[List[float] | None, str | None]:
        try:
            # Lazy import to avoid circular dependencies.
            from openai_clients import call_openai_embedding
        except Exception:
            return None, None

        embedding_payload = call_openai_embedding(text)
        if not embedding_payload or not isinstance(embedding_payload, dict):
            return None, None

        vector = self.normalize_vector(embedding_payload.get("embedding"))
        model = embedding_payload.get("model")
        return (vector if vector else None, model if model else None)


class HandlerStore:
    """Persist handlers to disk and expose a thread-safe cache."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.cache: Dict[str, Any] = {}
        self.mtime: float | None = None
        self.lock = RLock()

    def load(self) -> Dict[str, Any]:
        with self.lock:
            self._ensure_store_exists()

            try:
                mtime = self.path.stat().st_mtime
            except OSError as err:
                LOGGER.warning("Could not stat dynamic handler file: %s", err)
                return self._snapshot(self.cache)

            if self.mtime and self.mtime == mtime:
                return self._snapshot(self.cache)

            try:
                with self.path.open("r", encoding="utf-8") as file:
                    loaded = json.load(file)
                handlers_from_disk = loaded if isinstance(loaded, dict) else {}
                for handler in handlers_from_disk.values():
                    if isinstance(handler, dict):
                        handler["status"] = normalize_status(
                            handler.get("status"), default=_STATUS_APPROVED
                        )
                self.cache = self._snapshot(handlers_from_disk)
                self.mtime = mtime
            except json.JSONDecodeError:
                LOGGER.warning(
                    "dynamic_handlers.json is invalid JSON; keeping last good cache.",
                )
                self.mtime = mtime
            except OSError as err:
                LOGGER.warning("Failed to read dynamic handlers: %s", err)

            return self._snapshot(self.cache)

    def persist(self, handlers: Dict[str, Any]) -> None:
        with self.lock:
            temp_path = self.path.with_name(self.path.name + ".tmp")
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(handlers, file, indent=4)
            os.replace(temp_path, self.path)
            self.cache = self._snapshot(handlers)
            try:
                self.mtime = self.path.stat().st_mtime
            except OSError as err:
                LOGGER.debug("Could not refresh handler mtime after write: %s", err)
                self.mtime = None

    def _ensure_store_exists(self) -> None:
        if self.path.exists():
            return

        try:
            folder = self.path.parent
            if folder and str(folder) not in {"", "."}:
                folder.mkdir(parents=True, exist_ok=True)
            self.path.write_text("{}", encoding="utf-8")
            self.cache = {}
            self.mtime = self.path.stat().st_mtime
        except OSError as err:
            LOGGER.warning("Failed to initialize dynamic handler store: %s", err)

    @staticmethod
    def _snapshot(handlers: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(handlers) if isinstance(handlers, dict) else {}


class HandlerScorer:
    """Score handlers using embeddings, keywords, recency, and feedback."""

    def __init__(self, embeddings: EmbeddingService) -> None:
        self.embeddings = embeddings

    def keyword_match_score(
        self, prompt: str, handler_keywords: Set[str], entities: dict | None
    ) -> float:
        prompt_tokens = HandlerTextProcessor.extract_keywords(prompt)
        if not handler_keywords:
            return 0.0

        overlap = prompt_tokens.intersection(handler_keywords)
        token_balance = max(len(handler_keywords), len(prompt_tokens), 1)
        base_score = (len(overlap) * 1.35) / token_balance

        entity_bonus = 0.0
        if isinstance(entities, dict):
            prompt_lower = prompt.lower()
            for value in entities.values():
                if not value:
                    continue
                if str(value).lower() in prompt_lower:
                    entity_bonus += 0.08

        return min(base_score + entity_bonus, 1.25)

    def compute_handler_score(
        self,
        prompt: str,
        prompt_embedding: List[float] | None,
        handler_data: Dict[str, Any],
        handler_keywords: Set[str],
        handler_embedding: List[float] | None,
    ) -> float:
        keyword_score = self.keyword_match_score(
            prompt,
            handler_keywords,
            handler_data.get("entities")
            if isinstance(handler_data.get("entities"), dict)
            else {},
        )

        embedding_score = 0.0
        if prompt_embedding and handler_embedding:
            embedding_score = self.embeddings.cosine_similarity(
                prompt_embedding, handler_embedding
            )

        helpful = int(handler_data.get("helpful_count", 0) or 0)
        flagged = int(handler_data.get("flagged_count", 0) or 0)
        usage_count = int(handler_data.get("usage_count", 0) or 0)
        status = normalize_status(handler_data.get("status"), default=_STATUS_APPROVED)

        recency_bonus = 0.0
        recency_reference = (
            handler_data.get("last_used")
            or handler_data.get("last_updated")
            or handler_data.get("created_at")
        )
        if recency_reference:
            try:
                age_days = max((time.time() - float(recency_reference)) / 86400, 0)
                recency_bonus = max(0.12 - 0.03 * math.log1p(age_days), -0.05)
            except (TypeError, ValueError):
                recency_bonus = 0.0

        helpful_bonus = min(math.log1p(helpful) * 0.05, 0.35)
        usage_bonus = min(math.log1p(usage_count) * 0.03, 0.15)
        feedback_penalty = min(flagged * 0.07, 0.4)
        status_bonus = (
            0.06
            if status == _STATUS_APPROVED
            else (-0.02 if status == _STATUS_PENDING else -0.5)
        )

        composite = (0.65 * embedding_score) + (0.35 * keyword_score)
        composite += helpful_bonus + usage_bonus + recency_bonus + status_bonus
        composite -= feedback_penalty
        return composite


class DynamicHandlerService:
    """Orchestrate handler lifecycle operations."""

    def __init__(
        self,
        store: HandlerStore,
        embeddings: EmbeddingService,
        scorer: HandlerScorer,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.scorer = scorer

    def load_handlers(self) -> Dict[str, Any]:
        return self.store.load()

    def save_handler(self, name: str, handler_data: Dict[str, Any]) -> None:
        now = int(time.time())
        handler_snapshot = dict(handler_data or {})
        handler_snapshot["name"] = name
        handler_snapshot.setdefault(
            "last_prompt",
            handler_snapshot.get("prompt") or handler_snapshot.get("original_prompt"),
        )

        keywords = set(handler_snapshot.get("keywords") or [])
        keywords.update(HandlerTextProcessor.extract_keywords(name))

        prompt_text = handler_snapshot.get("prompt") or handler_snapshot.get(
            "original_prompt"
        )
        if prompt_text:
            keywords.update(HandlerTextProcessor.extract_keywords(prompt_text))

        entities = handler_snapshot.get("entities")
        if isinstance(entities, dict):
            keywords.update(
                HandlerTextProcessor.extract_keywords(
                    " ".join(str(value) for value in entities.values() if value is not None)
                )
            )

        handler_snapshot["keywords"] = sorted(keywords)

        with self.store.lock:
            handlers = self.store.load()
            existing = handlers.get(name) if isinstance(handlers.get(name), dict) else {}
            merged = dict(existing or {})
            merged.update(handler_snapshot)

            status_default = _STATUS_APPROVED if existing else _DEFAULT_STATUS
            merged["status"] = normalize_status(
                merged.get("status"), default=status_default
            )

            embedding_vector: List[float] | None = None
            embedding_model: str | None = None
            needs_embedding = not self.embeddings.normalize_vector(
                merged.get("embedding")
            )
            if needs_embedding:
                embed_text = HandlerTextProcessor.build_embedding_text(merged)
                if embed_text:
                    embedding_vector, embedding_model = self.embeddings.generate_handler_embedding(
                        embed_text
                    )

            merged.setdefault("created_at", now)
            merged["last_updated"] = now
            merged["usage_count"] = int(existing.get("usage_count", 0) or 0)
            merged["helpful_count"] = int(existing.get("helpful_count", 0) or 0)
            merged["flagged_count"] = int(existing.get("flagged_count", 0) or 0)

            if embedding_vector:
                merged["embedding"] = embedding_vector
                if embedding_model:
                    merged["embedding_model"] = embedding_model
                merged["last_embedded"] = now

            handlers[name] = merged

            try:
                self.store.persist(handlers)
            except OSError as err:
                LOGGER.warning("Failed to persist dynamic handler '%s': %s", name, err)

    def find_handler(self, prompt: str) -> Optional[Dict[str, Any]]:
        handlers = self.store.load()
        prompt_embedding = self.embeddings.get_prompt_embedding(prompt)
        requires_item_breakdown = HandlerTextProcessor.prompt_requires_item_breakdown(
            prompt
        )
        multi_item_prompt = HandlerTextProcessor.is_multi_item_usage_prompt(prompt)

        best_match = None
        highest_score = _FALLBACK_MATCH_THRESHOLD
        backfill_remaining = _MAX_EMBED_BACKFILL_PER_RUN

        for name, handler_data in handlers.items():
            if not isinstance(handler_data, dict):
                continue

            handler_data.setdefault("name", name)

            status = normalize_status(
                handler_data.get("status"), default=_STATUS_APPROVED
            )
            if status == _STATUS_DISABLED:
                continue

            helpful = int(handler_data.get("helpful_count", 0) or 0)
            flagged = int(handler_data.get("flagged_count", 0) or 0)
            if flagged >= 3 and flagged >= helpful + 1:
                continue

            if self._should_skip_for_prompt(
                prompt, handler_data, requires_item_breakdown, multi_item_prompt
            ):
                continue

            keywords = self._handler_keywords(handler_data, name)
            if not keywords or keywords.issubset({"custom", "sql", "query"}):
                continue

            handler_embedding = self.embeddings.normalize_vector(
                handler_data.get("embedding")
            )
            if prompt_embedding and not handler_embedding and backfill_remaining > 0:
                backfill_remaining -= 1
                backfilled = self._backfill_embedding(name, handler_data)
                if backfilled:
                    handler_embedding = self.embeddings.normalize_vector(
                        backfilled.get("embedding")
                    )
                    handlers[name] = backfilled

            score = self.scorer.compute_handler_score(
                prompt,
                prompt_embedding,
                handler_data,
                keywords,
                handler_embedding,
            )

            if score > highest_score:
                highest_score = score
                best_match = handler_data
                best_match["match_score"] = round(score, 3)

        if best_match:
            self._record_usage(best_match.get("name"))

        return best_match

    def record_feedback(self, handler_name: str, helpful: bool) -> bool:
        if not handler_name:
            return False

        with self.store.lock:
            handlers = self.store.load()
            handler = handlers.get(handler_name)
            if not isinstance(handler, dict):
                return False

            key = "helpful_count" if helpful else "flagged_count"
            handler[key] = int(handler.get(key, 0) or 0) + 1
            handler["last_feedback"] = int(time.time())

            if helpful and handler.get("status") == _STATUS_PENDING:
                handler["status"] = _STATUS_APPROVED

            if not helpful and int(handler.get("flagged_count", 0) or 0) >= 3:
                handler["status"] = _STATUS_DISABLED

            handler["last_updated"] = int(time.time())

            try:
                self.store.persist(handlers)
                return True
            except OSError as err:
                LOGGER.warning(
                    "Failed to record feedback for handler '%s': %s", handler_name, err
                )
                return False

    def update_status(
        self, handler_name: str, status: str, note: str | None = None
    ) -> bool:
        normalized = normalize_status(status, default=_DEFAULT_STATUS)
        if not handler_name:
            return False

        with self.store.lock:
            handlers = self.store.load()
            handler = handlers.get(handler_name)
            if not isinstance(handler, dict):
                return False

            handler["status"] = normalized
            handler["last_reviewed"] = int(time.time())
            handler["last_updated"] = int(time.time())
            if note is not None:
                handler["review_note"] = note.strip()

            try:
                self.store.persist(handlers)
                return True
            except OSError as err:
                LOGGER.warning(
                    "Failed to update status for handler '%s': %s", handler_name, err
                )
                return False

    def delete_handler(self, handler_name: str) -> bool:
        if not handler_name:
            return False

        with self.store.lock:
            handlers = self.store.load()
            if handler_name not in handlers:
                return False

            handlers.pop(handler_name, None)
            try:
                self.store.persist(handlers)
                return True
            except OSError as err:
                LOGGER.warning("Failed to delete handler '%s': %s", handler_name, err)
                return False

    def list_handlers(self) -> Dict[str, Any]:
        return self.store.load()

    def _should_skip_for_prompt(
        self,
        prompt: str,
        handler_data: Dict[str, Any],
        requires_item_breakdown: bool,
        multi_item_prompt: bool,
    ) -> bool:
        if requires_item_breakdown and not HandlerTextProcessor.sql_supports_item_breakdown(
            handler_data.get("sql")
        ):
            return True

        if HandlerTextProcessor.handler_targets_single_item(handler_data):
            if multi_item_prompt:
                return True

            item_hint = HandlerTextProcessor.extract_handler_item_code(handler_data)
            if item_hint and not HandlerTextProcessor.prompt_mentions_token(
                prompt, item_hint
            ):
                return True

        # Semantic mismatch check: if prompt uses specific logic/time terms,
        # the handler must contain relevant keywords or SQL symbols.
        logic_map = {
            "current": [("keywords_name", ["current", "curr", "now", "currcost"])],
            "standard": [("keywords_name", ["standard", "stnd", "stndcost"])],
            "last": [("keywords_name", ["last", "lst", "lstcost", "previous"])],
            "less than": [("sql", ["<", "where", "having"]), ("keywords_name", ["less", "lower", "below", "under"])],
            "greater than": [("sql", ["where", "having", ">"]), ("keywords_name", ["greater", "higher", "more", "above", "over"])],
            "compare": [("keywords_name", ["compare", "vs", "versus", "diff", "comparison"])],
            "difference": [("keywords_name", ["diff", "delta", "change", "variance"])],
            # For NOT and MISSING, avoid matching internal SQL patterns
            # Only match if keywords explicitly mention these concepts
            "not in": [("keywords_name", ["not", "exclude", "without", "except", "excluding"])],
            "missing": [("keywords_name", ["missing", "blank", "empty"])],
        }

        handler_name_keywords = (
            str(handler_data.get("name") or "")
            + " "
            + " ".join(str(k) for k in (handler_data.get("keywords") or []))
        ).lower()
        
        handler_sql = str(handler_data.get("sql") or "").lower()

        prompt_lower = prompt.lower()
        for trigger, checks in logic_map.items():
            if trigger in prompt_lower:
                # All check groups must pass at least one requirement
                handler_matches = False
                for check_type, required_any in checks:
                    if check_type == "keywords_name":
                        if any(req in handler_name_keywords for req in required_any):
                            handler_matches = True
                            break
                    elif check_type == "sql":
                        if all(req in handler_sql for req in required_any):
                            handler_matches = True
                            break
                
                if not handler_matches:
                    return True

        return False

    def _handler_keywords(self, handler_data: Dict[str, Any], fallback_name: str) -> Set[str]:
        keywords = set(handler_data.get("keywords") or [])
        if not keywords:
            fallback = (
                handler_data.get("prompt")
                or handler_data.get("name")
                or fallback_name
            )
            keywords.update(HandlerTextProcessor.extract_keywords(fallback))
        return keywords

    def _backfill_embedding(
        self, handler_name: str, handler_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        embed_text = HandlerTextProcessor.build_embedding_text(handler_data)
        if not embed_text:
            return None

        vector, model = self.embeddings.generate_handler_embedding(embed_text)
        if not vector:
            return None

        now = int(time.time())
        with self.store.lock:
            handlers = self.store.load()
            target = handlers.get(handler_name)
            if not isinstance(target, dict):
                return None

            target["embedding"] = vector
            if model:
                target["embedding_model"] = model
            target["last_embedded"] = now

            try:
                self.store.persist(handlers)
            except OSError as err:
                LOGGER.debug(
                    "Unable to persist embedding for handler '%s': %s", handler_name, err
                )
                return None
            return target

    def _record_usage(self, handler_name: str | None) -> None:
        if not handler_name:
            return

        with self.store.lock:
            handlers = self.store.load()
            target = handlers.get(handler_name)
            if not isinstance(target, dict):
                return

            target["usage_count"] = int(target.get("usage_count", 0) or 0) + 1
            target["last_used"] = int(time.time())
            if (
                target.get("status") == _STATUS_PENDING
                and target.get("usage_count", 0) >= 3
            ):
                target["status"] = _STATUS_APPROVED

            try:
                self.store.persist(handlers)
            except OSError:
                LOGGER.debug(
                    "Unable to update usage stats for handler '%s'", handler_name
                )


_STORE = HandlerStore(_DYNAMIC_HANDLERS_PATH)
_EMBEDDINGS = EmbeddingService()
_SCORER = HandlerScorer(_EMBEDDINGS)
_SERVICE = DynamicHandlerService(_STORE, _EMBEDDINGS, _SCORER)


def load_dynamic_handlers() -> Dict[str, Any]:
    """
    Load dynamic handlers with hot-reload support.
    Caches the last good snapshot but refreshes whenever the JSON file changes.
    """
    return _SERVICE.load_handlers()


def save_dynamic_handler(name: str, handler_data: Dict[str, Any]) -> None:
    """
    Save a new dynamic handler to the JSON file, enriching it with keywords for better recall.
    """
    _SERVICE.save_handler(name, handler_data)


def find_dynamic_handler(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Find a dynamic handler that matches the prompt based on keyword overlap and entities.
    """
    return _SERVICE.find_handler(prompt)


def record_handler_feedback(handler_name: str, helpful: bool) -> bool:
    """
    Update feedback counters for a learned handler so routing can prioritize good answers.
    Returns True when feedback was recorded.
    """
    return _SERVICE.record_feedback(handler_name, helpful)


def update_handler_status(handler_name: str, status: str, note: str | None = None) -> bool:
    """Set a handler's moderation status (approved/pending/disabled) for admin control."""
    return _SERVICE.update_status(handler_name, status, note)


def delete_dynamic_handler(handler_name: str) -> bool:
    """Hard delete a learned handler from the store."""
    return _SERVICE.delete_handler(handler_name)


def list_dynamic_handlers() -> Dict[str, Any]:
    """Expose learned handlers for UI display/management."""
    return _SERVICE.list_handlers()
