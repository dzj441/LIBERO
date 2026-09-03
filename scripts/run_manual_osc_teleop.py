#!/usr/bin/env python3
"""Browser-based human teleoperation over the benchmark's native OSC path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import signal
from typing import Any, Mapping
from urllib.parse import urlsplit

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero.agent_env import make_libero_agent_env
from libero.libero.agent_env.control import ActionInterface
from libero.libero.agent_env.private_recording import PrivateRolloutVideoRecorder
from libero.libero.agent_env.run_viewer import describe_server_urls
from libero.libero.agent_env.service import AgentEpisodeService


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_REQUEST_BYTES = 64 * 1024


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LIBERO Manual OSC Teleop</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; }
  body { margin: 20px auto; max-width: 1180px; padding: 0 18px; background:#111; color:#eee; }
  h1 { margin-bottom: 4px; }
  .sub { color:#aaa; margin-top:0; }
  .images { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
  figure { margin:0; background:#1c1c1c; border:1px solid #444; padding:8px; }
  figcaption { margin-bottom:7px; font-weight:650; }
  img { width:100%; aspect-ratio:1; object-fit:contain; image-rendering:auto; background:#000; }
  .panel { margin-top:14px; padding:14px; border:1px solid #444; background:#1c1c1c; }
  .row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:8px 0; }
  button { min-width:76px; padding:9px 12px; border:1px solid #666; border-radius:5px; background:#292929; color:#fff; cursor:pointer; }
  button:hover { background:#3b3b3b; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.danger { border-color:#a55; }
  button.active { border-color:#62b5ff; background:#164d73; }
  label { color:#ccc; }
  input { width:70px; padding:5px; background:#111; color:#fff; border:1px solid #555; }
  code, pre { background:#0b0b0b; }
  pre { min-height:90px; padding:10px; white-space:pre-wrap; overflow:auto; }
  .hint { color:#aaa; font-size:.92rem; }
  #status.good { color:#7ee787; }
  #status.bad { color:#ff7b72; }
  @media (max-width:850px) { .images { grid-template-columns:1fr; } }
</style>
</head>
<body>
<h1>LIBERO Manual OSC Teleop</h1>
<p class="sub">Arrange Table · EGL offscreen simulation · native OSC_POSE</p>
<div class="images">
  <figure><figcaption>Desired arrangement</figcaption><img id="goal" src="media/goal"></figure>
  <figure><figcaption>Head camera</figcaption><img id="head" src="media/head"></figure>
  <figure><figcaption>Wrist camera</figcaption><img id="wrist" src="media/wrist"></figure>
</div>

<div class="panel">
  <div class="row">
    <label>translation <input id="translation" type="number" min="0.01" max="1" step="0.05" value="0.35"></label>
    <label>rotation <input id="rotation" type="number" min="0.01" max="1" step="0.05" value="0.25"></label>
    <label>repeat cycles <input id="repeat" type="number" min="1" max="20" step="1" value="3"></label>
    <span id="status">loading…</span>
  </div>
  <div class="row"><strong>Translation:</strong>
    <button data-motion="x-">W · −X</button><button data-motion="x+">S · +X</button>
    <button data-motion="y-">A · −Y</button><button data-motion="y+">D · +Y</button>
    <button data-motion="z+">R · +Z</button><button data-motion="z-">F · −Z</button>
  </div>
  <div class="row"><strong>Rotation:</strong>
    <button data-motion="rx+">Z · +RX</button><button data-motion="rx-">X · −RX</button>
    <button data-motion="ry+">T · +RY</button><button data-motion="ry-">G · −RY</button>
    <button data-motion="rz+">C · +RZ</button><button data-motion="rz-">V · −RZ</button>
  </div>
  <div class="row"><strong>Persistent gripper drive:</strong>
    <button id="open">Open · −1</button><button id="neutral">Neutral · 0</button><button id="close">Close · +1</button>
    <span class="hint">Movement actions retain the selected drive; use +1 while carrying an object.</span>
  </div>
  <div class="row">
    <button id="refresh">Refresh images</button>
    <button id="finish" class="danger">Finish and run checker</button>
  </div>
  <p class="hint">Each click submits 1–20 identical normalized controller cycles. Translation 1.0 maps to a 5 cm OSC target and rotation 1.0 maps to 0.5 rad. Keys are ignored while an input field is focused.</p>
  <pre id="log"></pre>
</div>
<script>
let busy = false;
let gripper = -1;
let finished = false;
const log = document.getElementById('log');
const status = document.getElementById('status');

function bounded(id, lo, hi) {
  const value = Number(document.getElementById(id).value);
  if (!Number.isFinite(value)) throw new Error(id + ' must be finite');
  return Math.max(lo, Math.min(hi, value));
}
function setBusy(value) {
  busy = value;
  document.querySelectorAll('button').forEach(button => button.disabled = value || finished);
}
function setGrip(value) {
  gripper = value;
  for (const [id, candidate] of [['open',-1],['neutral',0],['close',1]]) {
    document.getElementById(id).classList.toggle('active', candidate === value);
  }
}
function refreshImages() {
  const nonce = Date.now();
  for (const id of ['goal','head','wrist']) document.getElementById(id).src = 'media/' + id + '?v=' + nonce;
}
async function post(path, payload={}) {
  if (busy || finished) return;
  setBusy(true);
  try {
    const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    const value = await response.json();
    log.textContent = JSON.stringify(value, null, 2);
    if (!response.ok || value.ok === false) throw new Error(value.error || 'request failed');
    status.textContent = value.observation_id || value.status || 'ready';
    status.className = 'good';
    if (path === 'api/finish') {
      finished = true;
      status.textContent = value.success ? 'checker: SUCCESS' : 'checker: FAILED';
      status.className = value.success ? 'good' : 'bad';
    }
    refreshImages();
  } catch (error) {
    status.textContent = error.message;
    status.className = 'bad';
  } finally {
    setBusy(false);
  }
}
async function move(name) {
  const t = bounded('translation', .01, 1);
  const r = bounded('rotation', .01, 1);
  const vector = [0,0,0,0,0,0,gripper];
  const map = { 'x-':[0,-t], 'x+':[0,t], 'y-':[1,-t], 'y+':[1,t], 'z-':[2,-t], 'z+':[2,t],
                'rx-':[3,-r], 'rx+':[3,r], 'ry-':[4,-r], 'ry+':[4,r], 'rz-':[5,-r], 'rz+':[5,r] };
  const [axis,value] = map[name]; vector[axis] = value;
  const repeat = Math.round(bounded('repeat', 1, 20));
  await post('api/action', {actions:Array.from({length:repeat}, () => [...vector])});
}
document.querySelectorAll('[data-motion]').forEach(button => button.onclick = () => move(button.dataset.motion));
document.getElementById('open').onclick = async () => { setGrip(-1); await moveGrip(); };
document.getElementById('neutral').onclick = () => setGrip(0);
document.getElementById('close').onclick = async () => { setGrip(1); await moveGrip(); };
async function moveGrip() {
  const repeat = Math.round(bounded('repeat', 1, 20));
  const vector = [0,0,0,0,0,0,gripper];
  await post('api/action', {actions:Array.from({length:repeat}, () => [...vector])});
}
document.getElementById('refresh').onclick = refreshImages;
document.getElementById('finish').onclick = () => { if (confirm('End this attempt and run the official checker?')) post('api/finish'); };
const keys = {'w':'x-','s':'x+','a':'y-','d':'y+','r':'z+','f':'z-','z':'rx+','x':'rx-','t':'ry+','g':'ry-','c':'rz+','v':'rz-'};
document.addEventListener('keydown', event => {
  if (event.repeat || ['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) return;
  const motion = keys[event.key.toLowerCase()];
  if (motion) { event.preventDefault(); move(motion); }
  if (event.code === 'Space') { event.preventDefault(); setGrip(gripper === 1 ? -1 : 1); moveGrip(); }
});
fetch('api/state').then(response => response.json()).then(value => {
  status.textContent = value.observation_id || 'ready'; status.className='good';
  log.textContent = JSON.stringify(value, null, 2); setGrip(-1); refreshImages();
});
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / "v1temp" / "arrange_table_teleop")
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=None,
        help="Optional total submission limit; manual teleoperation is unlimited by default",
    )
    return parser.parse_args()


def _new_run_directory(root: Path) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("manual_%Y%m%dT%H%M%SZ")
    for _ in range(20):
        candidate = root / f"{stamp}_{secrets.token_hex(2)}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a unique manual teleop directory")


def _json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("invalid Content-Length") from exc
    if length < 0 or length > MAX_REQUEST_BYTES:
        raise ValueError("request body is too large")
    body = handler.rfile.read(length)
    value = json.loads(body or b"{}")
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def _validated_actions(value: object) -> list[list[float]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 20:
        raise ValueError("actions must contain between 1 and 20 vectors")
    output: list[list[float]] = []
    for action in value:
        if not isinstance(action, list) or len(action) != 7:
            raise ValueError("every OSC action must be a seven-element array")
        try:
            vector = [float(component) for component in action]
        except (TypeError, ValueError) as exc:
            raise ValueError("every OSC component must be numeric") from exc
        if any(not math.isfinite(component) or abs(component) > 1 for component in vector):
            raise ValueError("every OSC component must be finite and within [-1, 1]")
        output.append(vector)
    return output


def _route_suffix(path: str, suffix: str) -> bool:
    """Accept direct routes and reverse-proxy-prefixed forms of the same route."""

    normalized = path.rstrip("/")
    return normalized == suffix or normalized.endswith(suffix)


class TeleopState:
    def __init__(self, service: AgentEpisodeService, run_directory: Path, workspace: Path) -> None:
        self.service = service
        self.run_directory = run_directory
        self.workspace = workspace
        self.last_response = service.handle({"command": "start"})

    @property
    def observation_directory(self) -> Path:
        return self.workspace / "benchmark_inputs" / "current_observation"

    def snapshot(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": self.service.state,
            "observation_id": self.service.latest_observation_id,
            "run_directory": os.fspath(self.run_directory),
            "max_agent_steps": self.service.agent_env.max_agent_steps,
        }

    def action(self, actions: object) -> dict[str, Any]:
        if self.service.latest_observation_id is None:
            raise RuntimeError("no current observation")
        self.last_response = self.service.handle(
            {
                "command": "osc_sequence",
                "observation_id": self.service.latest_observation_id,
                "actions": _validated_actions(actions),
            }
        )
        return self.last_response

    def finish(self) -> dict[str, Any]:
        if self.service.latest_observation_id is None:
            raise RuntimeError("no current observation")
        self.last_response = self.service.handle(
            {
                "command": "finish",
                "observation_id": self.service.latest_observation_id,
            }
        )
        return self.last_response


def make_handler(state: TeleopState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "LiberoManualTeleop/1"

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[teleop] {self.address_string()} {fmt % args}", flush=True)

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if _route_suffix(path, "/api/state"):
                return self._json(state.snapshot())
            media = None
            for suffix, candidate in {
                "/media/head": state.observation_directory / "head" / "rgb.png",
                "/media/wrist": state.observation_directory / "wrist" / "rgb.png",
                "/media/goal": state.observation_directory / "task_reference" / "rgb.png",
            }.items():
                if _route_suffix(path, suffix):
                    media = candidate
                    break
            if media is None and path.endswith("/"):
                return self._bytes(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if media is None:
                return self._error(HTTPStatus.NOT_FOUND, "unknown route")
            if not media.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "image is unavailable")
            return self._bytes(media.read_bytes(), mimetypes.guess_type(media.name)[0] or "application/octet-stream")

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            try:
                body = _json_body(self)
                if _route_suffix(path, "/api/action"):
                    response = state.action(body.get("actions"))
                elif _route_suffix(path, "/api/finish"):
                    response = state.finish()
                else:
                    return self._error(HTTPStatus.NOT_FOUND, "unknown route")
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return self._json(response)

        def _json(self, value: Mapping[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            self._bytes(json.dumps(dict(value), indent=2, sort_keys=True).encode("utf-8"), "application/json", status)

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._json({"ok": False, "error": message}, status)

        def _bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main() -> int:
    args = parse_args()
    run_directory = _new_run_directory(args.output_root)
    workspace = run_directory / "workspace"
    workspace.mkdir()
    recorder = PrivateRolloutVideoRecorder(run_directory / "continuous_video.mp4")
    agent_env = make_libero_agent_env(
        suite="libero_arrange_table",
        task_id=0,
        init_state_id=args.init_state_id,
        profile="level4",
        seed=args.seed,
        camera_height=args.resolution,
        camera_width=args.resolution,
        render_gpu_device_id=args.render_gpu_device_id,
        max_agent_steps=args.max_agent_steps,
        private_control_step_callback=recorder.append_raw_observation,
    )
    service = AgentEpisodeService(
        agent_env,
        workspace_directory=workspace,
        current_observation_directory=workspace / "benchmark_inputs" / "current_observation",
        private_run_directory=run_directory,
        action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
    )
    state: TeleopState | None = None
    server: HTTPServer | None = None
    try:
        state = TeleopState(service, run_directory, workspace)
        server = HTTPServer((args.host, args.port), make_handler(state))
        actual_port = int(server.server_address[1])
        print("LIBERO manual OSC teleop is ready.", flush=True)
        for url in describe_server_urls(args.host, actual_port):
            print(url, flush=True)
        print(f"Run artifacts: {run_directory}", flush=True)

        def stop(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        if not service.finished:
            service.finalize_aborted("manual_teleop_stopped_before_finish")
        service.close()
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
