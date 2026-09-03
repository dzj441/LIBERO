"""Unit tests for the headless manual OSC teleoperation boundary."""

import pytest

from scripts.run_manual_osc_teleop import PAGE, _route_suffix, _validated_actions, parse_args


def test_manual_teleop_accepts_one_to_twenty_normalized_actions():
    action = [0.0, 0.25, -1.0, 0.5, 0.0, 0.0, 1.0]
    assert _validated_actions([action] * 20) == [action] * 20


@pytest.mark.parametrize(
    "actions, message",
    [
        ([], "between 1 and 20"),
        ([[0.0] * 7] * 21, "between 1 and 20"),
        ([[0.0] * 6], "seven-element"),
        ([[0.0] * 6 + [1.01]], r"within \[-1, 1\]"),
        ([[0.0] * 6 + [float("nan")]], "finite"),
        ([[0.0] * 6 + [None]], "numeric"),
    ],
)
def test_manual_teleop_rejects_invalid_actions(actions, message):
    with pytest.raises(ValueError, match=message):
        _validated_actions(actions)


def test_manual_teleop_page_exposes_goal_head_wrist_and_finish():
    assert 'src="media/goal"' in PAGE
    assert 'src="media/head"' in PAGE
    assert 'src="media/wrist"' in PAGE
    assert "post('api/action'" in PAGE
    assert "post('api/finish'" in PAGE


def test_manual_teleop_routes_accept_vscode_proxy_prefix():
    prefix = "/workspace/vscode/session/proxy/8766"
    assert _route_suffix(f"{prefix}/api/state", "/api/state")
    assert _route_suffix(f"{prefix}/media/head", "/media/head")
    assert not _route_suffix(f"{prefix}/media/head-extra", "/media/head")


def test_manual_teleop_has_no_total_action_budget_by_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_manual_osc_teleop.py"])
    assert parse_args().max_agent_steps is None
