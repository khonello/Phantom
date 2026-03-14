# This Python file uses the following encoding: utf-8
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterType

from desktop.controller import PipelineClient
from desktop.bridge import Bridge, FrameDisplay


def main() -> None:
    parser = argparse.ArgumentParser(description='roop-cam desktop — GUI controller for the pipeline')
    parser.add_argument('--host', default='localhost', help='pipeline host (default: localhost)')
    parser.add_argument('--port', type=int, default=9000, help='pipeline control port (default: 9000)')
    args = parser.parse_args()

    app = QApplication(sys.argv)

    client = PipelineClient(args.host, args.port)
    bridge = Bridge(client)

    qmlRegisterType(FrameDisplay, 'Phantom', 1, 0, 'FrameDisplay')

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty('bridge', bridge)

    qml_file = Path(__file__).resolve().parent / 'main.qml'
    engine.load(qml_file)
    if not engine.rootObjects():
        sys.exit(-1)

    result = app.exec()
    bridge.cleanup()
    sys.exit(result)


if __name__ == '__main__':
    main()
