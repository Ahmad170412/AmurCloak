"""The ``amurcloak`` command - the friendly front door.

Works out which port to use, notices if a copy is already running (and re-opens
the browser instead of starting a second server on a port nobody asked for),
opens the browser, and hands off to the Flask app.

Local only. The one request this module makes is a loopback probe of 127.0.0.1
to see whether AmurCloak is already there. It cannot leave the machine, and it
deliberately ignores any HTTP proxy configured in the environment, so a
misconfigured proxy cannot turn a localhost check into an outbound one.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
import webbrowser

from .app import BIND_FAILED, parse_args, run_app

PORT_START = 5000
PORT_TRIES = 20
PROBE_TIMEOUT = 0.35

# Loopback probes must never be routed through a proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _is_free(port: int) -> bool:
    """True when 127.0.0.1:port can be bound.

    A bind test rather than a connect test, because binding is what the server
    actually attempts, so this answers the real question.

    SO_REUSEADDR is set to match what Werkzeug sets when it binds, so this
    predicts the real bind instead of being stricter than it. Without it, a
    socket merely lingering in TIME_WAIT reads as a busy port and gets skipped
    even though the server could have gone ahead and used it. It does not let
    us steal a port from a live listener -- that would need SO_REUSEPORT.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _is_running(port: int) -> bool:
    """True when *our* server answers on this port, not some other program."""
    try:
        with _OPENER.open(f"http://127.0.0.1:{port}/api/config", timeout=PROBE_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return False
    # Both keys are always present in our /api/config, and neither is likely on
    # an unrelated service that happens to share the port.
    return isinstance(payload, dict) and {"supported", "limits"} <= set(payload)


def _find_running(explicit: int | None) -> int | None:
    candidates = (
        [explicit] if explicit is not None else range(PORT_START, PORT_START + PORT_TRIES)
    )
    for port in candidates:
        if _is_running(port):
            return port
    return None


def _find_free_port() -> int:
    for port in range(PORT_START, PORT_START + PORT_TRIES):
        if _is_free(port):
            return port
    raise SystemExit(
        f"\n  No free port between {PORT_START} and {PORT_START + PORT_TRIES - 1}.\n"
        "  Choose one yourself:  amurcloak --port 8080\n"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    args.open = not args.no_open

    # Already running? Bring the tab back. Running the command twice must never
    # strand someone with a second server they did not ask for.
    running = _find_running(args.port)
    if running is not None:
        url = f"http://localhost:{running}"
        print("\n  AmurCloak is already running.")
        print(f"  {url}\n")
        if args.open:
            webbrowser.open(url)
        return 0

    explicit = args.port is not None
    if explicit:
        port = args.port
        # A port you named yourself is a promise, so fail clearly rather than
        # quietly serving on some other one.
        if not _is_free(port):
            raise SystemExit(
                f"\n  Port {port} is already in use by something else.\n"
                "  Pick another, or drop --port and let AmurCloak choose one.\n"
            )
    else:
        port = _find_free_port()
        if port != PORT_START:
            print(f"\n  Port {PORT_START} is busy, so AmurCloak will use {port}.")

    # There is an unavoidable gap between choosing a port and binding it. If
    # something slips into that gap, move to the next port instead of showing a
    # failure: the whole point of this command is that nobody has to think
    # about ports. Bounded, in case a thief keeps taking the port we pick.
    for _ in range(4):
        args.port = port
        code = run_app(args)
        if code != BIND_FAILED or explicit:
            return code
        port = _find_free_port()
        print(f"  Retrying on port {port}.\n")

    print("  Could not find a port that stays free. Try:  amurcloak --port 8080\n")
    return 1
