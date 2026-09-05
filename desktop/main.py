# This Python file uses the following encoding: utf-8
import sys
import argparse

from dotenv import load_dotenv
load_dotenv()

from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType

from desktop.controller import PipelineClient
from desktop.bridge import Bridge, FrameDisplay
from desktop.resources import resource_path


def main() -> None:
    parser = argparse.ArgumentParser(description='roop-cam desktop — GUI controller for the pipeline')
    parser.add_argument('--host', default='localhost', help='pipeline host (default: localhost)')
    parser.add_argument('--port', type=int, default=9000, help='pipeline control port (default: 9000)')
    args = parser.parse_args()

    app = QApplication(sys.argv)

    client = PipelineClient(args.host, args.port)
    bridge = Bridge(client)

    # The QML name is a str at runtime; PySide6 6.10's stubs type that
    # parameter as bytes. Passing bytes would register a name QML cannot
    # resolve, so the stub is what is wrong here, not the call.
    qmlRegisterType(FrameDisplay, 'Phantom', 1, 0, 'FrameDisplay')  # type: ignore[call-overload]

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty('bridge', bridge)

    # Resolved rather than assumed: in a standalone build main.qml is a data
    # file, and a build that lost it otherwise exits -1 with nothing said.
    qml_file = resource_path('main.qml')
    engine.load(qml_file)
    if not engine.rootObjects():
        print('Failed to load {} — QML did not produce a root object.'.format(qml_file),
              file=sys.stderr)
        sys.exit(-1)

    result = app.exec()
    bridge.cleanup()
    sys.exit(result)


if __name__ == '__main__':
    main()
