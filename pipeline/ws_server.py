import threading
from typing import Set

from websockets.sync.server import ServerConnection, serve

NAME = 'ROOP.WS'

_clients: Set[ServerConnection] = set()
_lock = threading.Lock()


def broadcast(jpeg_bytes: bytes) -> None:
    with _lock:
        dead: Set[ServerConnection] = set()
        for client in _clients:
            try:
                client.send(jpeg_bytes)
            except Exception:
                dead.add(client)
        _clients.difference_update(dead)


def _handler(conn: ServerConnection) -> None:
    with _lock:
        _clients.add(conn)
    try:
        while True:
            conn.recv()  # blocks; raises on disconnect
    except Exception:
        pass
    finally:
        with _lock:
            _clients.discard(conn)


def start(port: int = 9001) -> threading.Thread:
    server = serve(_handler, '0.0.0.0', port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f'[{NAME}] Frame WebSocket on ws://0.0.0.0:{port}')
    return thread
