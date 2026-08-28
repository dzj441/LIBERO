import json
import subprocess
import sys
from pathlib import Path

from scripts.libero_mcp_server import (
    MCP_PROTOCOL_VERSION,
    SERVER_INSTRUCTIONS,
    handle_message,
    request_for_tool,
    tool_definitions,
)


def test_mcp_initialize_and_tools_are_minimal():
    initialized = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": MCP_PROTOCOL_VERSION},
        }
    )
    assert initialized["result"]["instructions"] == SERVER_INSTRUCTIONS
    assert initialized["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION

    listed = handle_message(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )
    tools = listed["result"]["tools"]
    assert [tool["name"] for tool in tools] == [
        "start_episode",
        "osc_sequence",
        "finish_episode",
    ]
    vector_schema = tool_definitions()[1]["inputSchema"]["properties"]["actions"][
        "items"
    ]
    assert vector_schema["minItems"] == vector_schema["maxItems"] == 7


def test_mcp_tool_requests_map_to_existing_service_wire_protocol():
    assert request_for_tool("start_episode", {}) == {"command": "start"}
    actions = [[0.0, 0.1, 0.0, 0.0, 0.0, -0.2, 1.0]]
    assert request_for_tool("osc_sequence", {"actions": actions}) == {
        "command": "osc_sequence",
        "actions": actions,
    }
    assert request_for_tool("finish_episode", {}) == {"command": "finish"}


def test_stdio_adapter_speaks_newline_delimited_jsonrpc():
    script = Path(__file__).resolve().parents[2] / "scripts/libero_mcp_server.py"
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": MCP_PROTOCOL_VERSION},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    completed = subprocess.run(
        [sys.executable, str(script)],
        input="".join(json.dumps(message) + "\n" for message in messages),
        text=True,
        capture_output=True,
        check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2]
    assert responses[1]["result"]["tools"][1]["name"] == "osc_sequence"
