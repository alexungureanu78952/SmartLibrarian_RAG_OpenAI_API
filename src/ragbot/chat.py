"""Core chat orchestration for recommendation + summary tool calling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from ragbot.config import Settings
from ragbot.data_loader import build_summary_dict, load_book_entries
from ragbot.openai_retry import call_with_retry
from ragbot.retriever import COLLECTION_NAME, Retriever, collection_name_for_embedding
from ragbot.tools import RESPONSES_TOOL_SPEC, SummaryTool


@dataclass
class ChatResult:
    """Final answer payload returned by the chatbot orchestrator."""

    title: str
    reason: str
    full_summary: str
    retrieval_hits: list[dict[str, str]]


class BookChatbot:
    """Book chatbot that combines retrieval and function calling."""

    def __init__(self, settings: Settings):
        """Initialize OpenAI client, summary tool, and retriever."""
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        entries = load_book_entries(settings.summaries_json)
        summary_dict = build_summary_dict(entries)
        self.summary_tool = SummaryTool(summary_dict)
        self.retrievers = self._build_retrievers()

    def _build_retrievers(self) -> list[Retriever]:
        """Create retrievers in priority order for primary + fallback embedding models."""
        models: list[str] = []
        for model in [self.settings.embed_model, *self.settings.embed_fallback_models]:
            if model and model not in models:
                models.append(model)

        retrievers: list[Retriever] = []
        for model in models:
            collection_names = [collection_name_for_embedding(model)]
            if model == self.settings.embed_model:
                # Compatibility path for projects previously indexed into the legacy collection name.
                collection_names.append(COLLECTION_NAME)

            for collection_name in collection_names:
                retrievers.append(
                    Retriever.from_paths(
                        openai_client=self.client,
                        chroma_dir=str(self.settings.chroma_dir),
                        embedding_model=model,
                        collection_name=collection_name,
                    )
                )
        return retrievers

    def _search_with_fallbacks(self, user_query: str) -> list[dict[str, str]]:
        """Try retrievers in order and return first non-empty hit set."""
        errors: list[str] = []
        for retriever in self.retrievers:
            try:
                hits = retriever.search(user_query, top_k=self.settings.top_k)
                if hits:
                    return hits
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{retriever.collection_name}:{exc}")

        if errors:
            raise ValueError("Retrieval failed across all configured models/collections.")
        return []

    def _recommend_title(self, user_query: str, hits: list[dict[str, str]]) -> tuple[str, str]:
        """Use LLM to choose one title from retrieved context."""
        context_lines = []
        for i, hit in enumerate(hits, start=1):
            context_lines.append(
                f"{i}. Title: {hit['title']} | Themes: {hit['themes']} | Context: {hit['summary']}"
            )

        resp = call_with_retry(
            lambda: self.client.responses.create(
                model=self.settings.chat_model,
                instructions=(
                    "You are a helpful book recommendation assistant. "
                    "Recommend exactly one title from the provided context. "
                    "Return only a valid JSON object with keys: title, reason."
                ),
                input=[
                    {
                        "role": "user",
                        "content": (
                            f"User request: {user_query}\n\n"
                            f"Retrieved context:\n" + "\n".join(context_lines)
                        ),
                    }
                ],
            )
        )

        data = self._parse_json_object(resp.output_text or "{}")
        title = str(data.get("title", "")).strip()
        reason = str(data.get("reason", "")).strip()
        if not title:
            raise ValueError("Model did not return a title.")
        if not reason:
            reason = "This title best matches your interests based on the retrieved themes."
        return title, reason

    def _call_summary_tool_via_llm(self, title: str) -> str:
        """Force a call to get_summary_by_title and return final natural response."""
        first = call_with_retry(
            lambda: self.client.responses.create(
                model=self.settings.chat_model,
                instructions="You must call get_summary_by_title with the exact provided title.",
                input=[
                    {
                        "role": "user",
                        "content": f"Call get_summary_by_title for this exact title: {title}",
                    }
                ],
                tools=RESPONSES_TOOL_SPEC,
                tool_choice={"type": "function", "name": "get_summary_by_title"},
            )
        )

        tool_calls = self._extract_function_calls(first)
        if not tool_calls:
            print("[DEBUG] Tool call missing: using local fallback summary lookup.")
            return self.summary_tool.get_summary_by_title(title)

        print("[DEBUG] Tool call detected: get_summary_by_title invoked via model tool call.")
        tool_call = tool_calls[0]
        try:
            arguments = json.loads(tool_call["arguments"])
        except json.JSONDecodeError:
            arguments = {"title": title}
        tool_title = str(arguments.get("title", title))
        tool_result = self.summary_tool.get_summary_by_title(tool_title)

        follow_up_input = self._output_items_to_input(first.output)
        follow_up_input.append(
            {
                "type": "function_call_output",
                "call_id": tool_call["call_id"],
                "output": tool_result,
            }
        )

        second = call_with_retry(
            lambda: self.client.responses.create(
                model=self.settings.chat_model,
                instructions="Respond with recommendation and the provided full summary.",
                input=follow_up_input,
                tools=RESPONSES_TOOL_SPEC,
            )
        )

        final_text = second.output_text or tool_result
        return final_text

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        """Parse JSON object from model text with a safe fallback strategy."""
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _extract_function_calls(response: Any) -> list[dict[str, str]]:
        """Extract function calls from Responses API output items."""
        calls: list[dict[str, str]] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            calls.append(
                {
                    "call_id": str(getattr(item, "call_id", "")),
                    "name": str(getattr(item, "name", "")),
                    "arguments": str(getattr(item, "arguments", "{}")),
                }
            )
        return calls

    @staticmethod
    def _output_items_to_input(output_items: list[Any]) -> list[Any]:
        """Convert SDK output items to input list accepted by Responses API."""
        normalized: list[Any] = []
        for item in output_items:
            if hasattr(item, "model_dump"):
                normalized.append(item.model_dump())
            else:
                normalized.append(item)
        return normalized

    def ask(self, user_query: str) -> ChatResult:
        """Run full flow: retrieve -> recommend -> fetch full summary."""
        hits = self._search_with_fallbacks(user_query)
        if not hits:
            raise ValueError("No retrieval hits found. Run indexing first.")

        title, reason = self._recommend_title(user_query, hits)
        full_summary = self._call_summary_tool_via_llm(title)

        return ChatResult(
            title=title,
            reason=reason,
            full_summary=full_summary,
            retrieval_hits=hits,
        )
