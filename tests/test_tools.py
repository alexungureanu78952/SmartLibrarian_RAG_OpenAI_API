from ragbot.tools import RESPONSES_TOOL_SPEC, SummaryTool


def test_get_summary_by_title_exact_case_insensitive() -> None:
    tool = SummaryTool({"The Hobbit": "A fantasy adventure."})
    assert tool.get_summary_by_title("the hobbit") == "A fantasy adventure."


def test_get_summary_by_title_missing() -> None:
    tool = SummaryTool({"1984": "Dystopia."})
    result = tool.get_summary_by_title("Unknown")
    assert "could not find an exact match" in result.lower()


def test_tool_schema_is_strict() -> None:
    function_def = RESPONSES_TOOL_SPEC[0]
    assert function_def["strict"] is True
    assert function_def["parameters"]["additionalProperties"] is False
