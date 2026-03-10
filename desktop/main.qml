import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import Phantom 1.0

Window {
    id: root
    visible: true
    width: 1440
    height: 600
    minimumWidth: 900
    minimumHeight: 600
    title: "Phantom"
    color: "#09090e"

    // ── Header ────────────────────────────────────────────────────────
    Rectangle {
        id: header
        anchors { top: parent.top; left: parent.left; right: parent.right }
        height: 52
        color: "transparent"

        Rectangle {
            anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
            height: 1; color: "#14142a"
        }

        Row {
            anchors { left: parent.left; leftMargin: 24; verticalCenter: parent.verticalCenter }
            spacing: 12

            Rectangle {
                width: 9; height: 9; radius: 2
                anchors.verticalCenter: parent.verticalCenter
                gradient: Gradient {
                    orientation: Gradient.Horizontal
                    GradientStop { position: 0.0; color: "#8b5cf6" }
                    GradientStop { position: 1.0; color: "#3b82f6" }
                }
            }
            Text {
                text: "PHANTOM"
                color: "#e2e8f0"; font.pixelSize: 12
                font.letterSpacing: 4; font.weight: Font.Medium
                anchors.verticalCenter: parent.verticalCenter
            }
        }

        Row {
            anchors { right: parent.right; rightMargin: 24; verticalCenter: parent.verticalCenter }
            spacing: 16

            Text {
                text: bridge.statusMessage
                color: "#334155"; font.pixelSize: 12
                anchors.verticalCenter: parent.verticalCenter
            }

            Rectangle { width: 1; height: 18; color: "#14142a"; anchors.verticalCenter: parent.verticalCenter }

            Row {
                spacing: 8; anchors.verticalCenter: parent.verticalCenter

                Rectangle {
                    width: 7; height: 7; radius: 3.5
                    color: bridge.connected ? "#10b981" : "#ef4444"
                    anchors.verticalCenter: parent.verticalCenter

                    SequentialAnimation on opacity {
                        running: bridge.connected; loops: Animation.Infinite
                        NumberAnimation { to: 0.3; duration: 1100; easing.type: Easing.InOutSine }
                        NumberAnimation { to: 1.0; duration: 1100; easing.type: Easing.InOutSine }
                    }
                }
                Text {
                    text: bridge.connectionLabel
                    color: bridge.connected ? "#475569" : "#ef4444"
                    font.pixelSize: 12; anchors.verticalCenter: parent.verticalCenter
                }
            }
        }
    }

    // ── Body ──────────────────────────────────────────────────────────
    Item {
        anchors { top: header.bottom; bottom: parent.bottom; left: parent.left; right: parent.right }

        // ── Left sidebar ──────────────────────────────────────────────
        Rectangle {
            id: sidebar
            anchors { top: parent.top; bottom: parent.bottom; left: parent.left }
            width: 256
            color: "#0d0d18"

            Rectangle {
                anchors { top: parent.top; bottom: parent.bottom; right: parent.right }
                width: 1; color: "#14142a"
            }

            ColumnLayout {
                anchors { fill: parent; margins: 20; bottomMargin: 20 }
                spacing: 0

                // ── Face source ───────────────────────────────────────
                Text {
                    text: "FACE SOURCE"
                    color: "#252545"; font.pixelSize: 8; font.letterSpacing: 1.5
                    Layout.bottomMargin: 8
                }

                // ── Select button (no source set, or embedding in progress) ──
                Rectangle {
                    id: faceBtn
                    Layout.fillWidth: true; height: 38; radius: 8
                    visible: !bridge.sourceSet || bridge.embeddingPending
                    color: faceHover.containsMouse && !bridge.embeddingPending ? "#1a1a2e" : "#12121e"
                    border.color: faceHover.containsMouse && !bridge.embeddingPending ? "#2e2e50" : "#1e1e35"
                    border.width: 1
                    Behavior on color      { ColorAnimation { duration: 130 } }
                    Behavior on border.color { ColorAnimation { duration: 130 } }

                    Row {
                        anchors.centerIn: parent; spacing: 8

                        Rectangle {
                            width: 20; height: 20; radius: 10
                            color: bridge.embeddingPending ? "#1a1a2e" : "#1a0a35"
                            anchors.verticalCenter: parent.verticalCenter
                            Text {
                                anchors.centerIn: parent
                                text: bridge.embeddingPending ? "·" : "+"
                                color: bridge.embeddingPending ? "#475569" : "#8b5cf6"
                                font.pixelSize: bridge.embeddingPending ? 22 : 17
                            }
                        }
                        Text {
                            text: bridge.embeddingPending ? "processing…" : "Select Source Images"
                            color: bridge.embeddingPending ? "#475569" : "#a78bfa"
                            font.pixelSize: 12; anchors.verticalCenter: parent.verticalCenter
                        }
                    }

                    HoverHandler { id: faceHover }
                    MouseArea {
                        anchors.fill: parent; enabled: !bridge.embeddingPending
                        onClicked: bridge.selectFaceImages()
                        cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                    }
                }

                // ── Thumbnail card (source set, not processing) ────────
                Rectangle {
                    id: faceThumbnailCard
                    Layout.fillWidth: true; height: 76; radius: 8
                    visible: bridge.sourceSet && !bridge.embeddingPending
                    color: "#12121e"
                    border.color: "#2a1a45"; border.width: 1
                    clip: true

                    Image {
                        anchors.fill: parent
                        source: bridge.sourceThumbnail !== ""
                                ? "file:///" + bridge.sourceThumbnail
                                : ""
                        fillMode: Image.PreserveAspectCrop
                        smooth: true
                    }

                    // bottom label bar
                    Rectangle {
                        anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
                        height: 26
                        color: "#d009090e"

                        Text {
                            anchors { left: parent.left; leftMargin: 10; verticalCenter: parent.verticalCenter }
                            text: bridge.sourceLabel
                            color: "#c4b5fd"; font.pixelSize: 10
                            elide: Text.ElideMiddle
                            width: parent.width - 20
                        }
                    }

                    // × reset button (top-right)
                    Rectangle {
                        anchors { top: parent.top; right: parent.right; topMargin: 6; rightMargin: 6 }
                        width: 22; height: 22; radius: 5
                        color: resetHover.containsMouse ? "#3b1d6e" : "#1a0a35"
                        border.color: resetHover.containsMouse ? "#6d28d9" : "#2d1a45"
                        border.width: 1
                        Behavior on color { ColorAnimation { duration: 120 } }

                        Text {
                            anchors.centerIn: parent
                            text: "×"; color: "#a78bfa"; font.pixelSize: 13
                        }

                        HoverHandler { id: resetHover }
                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: bridge.resetSource()
                        }
                    }
                }

                // ── Divider ───────────────────────────────────────────
                Rectangle {
                    Layout.fillWidth: true; height: 1; color: "#14142a"
                    Layout.topMargin: 20; Layout.bottomMargin: 20
                }

                // ── Webcam ────────────────────────────────────────────
                Text {
                    text: "WEBCAM INDEX"
                    color: "#252545"; font.pixelSize: 8; font.letterSpacing: 1.5
                    Layout.bottomMargin: 8
                }

                Rectangle {
                    Layout.fillWidth: true; height: 38; radius: 8
                    color: "#12121e"
                    border.color: wcInput.activeFocus ? "#3a3a60" : "#1e1e35"
                    border.width: 1
                    Behavior on border.color { ColorAnimation { duration: 150 } }

                    TextInput {
                        id: wcInput
                        anchors { fill: parent; leftMargin: 14; rightMargin: 14 }
                        verticalAlignment: TextInput.AlignVCenter
                        text: "0"; color: "#e2e8f0"; font.pixelSize: 13
                        onEditingFinished: bridge.setWebcamIndex(text)
                        validator: IntValidator { bottom: 0; top: 9 }
                    }
                }

                // ── Quality ───────────────────────────────────────────
                Text {
                    text: "QUALITY"
                    color: "#252545"; font.pixelSize: 8; font.letterSpacing: 1.5
                    Layout.topMargin: 16; Layout.bottomMargin: 8
                }

                Rectangle {
                    id: qualBox
                    Layout.fillWidth: true; height: 38; radius: 8
                    color: qualHover.containsMouse ? "#1a1a2e" : "#12121e"
                    border.color: qualBox.open ? "#3a3a60" : "#1e1e35"
                    border.width: 1
                    z: open ? 10 : 0
                    Behavior on color { ColorAnimation { duration: 130 } }

                    property var opts: ["fast", "optimal", "production"]
                    property int sel: 1
                    property bool open: false

                    Row {
                        anchors { fill: parent; leftMargin: 14; rightMargin: 10 }
                        Text {
                            text: qualBox.opts[qualBox.sel]
                            color: "#cbd5e1"; font.pixelSize: 12
                            width: parent.width - 20
                            anchors.verticalCenter: parent.verticalCenter
                        }
                        Text { text: "⌄"; color: "#334155"; font.pixelSize: 10; anchors.verticalCenter: parent.verticalCenter }
                    }

                    HoverHandler { id: qualHover }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: qualBox.open = !qualBox.open
                        cursorShape: Qt.PointingHandCursor
                    }

                    Rectangle {
                        visible: qualBox.open
                        anchors.top: parent.bottom; anchors.topMargin: 4
                        anchors.left: parent.left
                        width: parent.width
                        height: qualBox.opts.length * 32 + 10
                        radius: 8; color: "#12121e"
                        border.color: "#252545"; border.width: 1

                        Column {
                            anchors { fill: parent; margins: 5 }
                            spacing: 2

                            Repeater {
                                model: qualBox.opts
                                Rectangle {
                                    width: parent.width; height: 30; radius: 5
                                    color: qualBox.sel === index ? "#1e1e38"
                                         : (rh.containsMouse ? "#171730" : "transparent")
                                    HoverHandler { id: rh }
                                    Text {
                                        anchors { left: parent.left; leftMargin: 10; verticalCenter: parent.verticalCenter }
                                        text: modelData
                                        color: qualBox.sel === index ? "#c4b5fd" : "#475569"
                                        font.pixelSize: 12
                                    }
                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: {
                                            qualBox.sel = index
                                            bridge.setQuality(modelData)
                                            qualBox.open = false
                                        }
                                        cursorShape: Qt.PointingHandCursor
                                    }
                                }
                            }
                        }
                    }
                }

                // ── Platform ──────────────────────────────────────────
                Text {
                    text: "PLATFORM"
                    color: "#252545"; font.pixelSize: 8; font.letterSpacing: 1.5
                    Layout.topMargin: 16; Layout.bottomMargin: 8
                }

                Rectangle {
                    id: platBox
                    Layout.fillWidth: true; height: 38; radius: 8
                    color: platHover.containsMouse ? "#1a1a2e" : "#12121e"
                    border.color: platBox.open ? "#3a3a60" : "#1e1e35"
                    border.width: 1
                    z: open ? 10 : 0
                    Behavior on color { ColorAnimation { duration: 130 } }

                    property var opts: ["obs", "unitycapture"]
                    property var labels: ["OBS Virtual Camera", "Unity Capture"]
                    property int sel: 0
                    property bool open: false

                    Row {
                        anchors { fill: parent; leftMargin: 14; rightMargin: 10 }
                        Text {
                            text: platBox.labels[platBox.sel]
                            color: "#cbd5e1"; font.pixelSize: 12
                            width: parent.width - 20
                            anchors.verticalCenter: parent.verticalCenter
                            elide: Text.ElideRight
                        }
                        Text { text: "⌄"; color: "#334155"; font.pixelSize: 10; anchors.verticalCenter: parent.verticalCenter }
                    }

                    HoverHandler { id: platHover }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: platBox.open = !platBox.open
                        cursorShape: Qt.PointingHandCursor
                    }

                    Rectangle {
                        visible: platBox.open
                        anchors.top: parent.bottom; anchors.topMargin: 4
                        anchors.left: parent.left
                        width: parent.width
                        height: platBox.opts.length * 32 + 10
                        radius: 8; color: "#12121e"
                        border.color: "#252545"; border.width: 1

                        Column {
                            anchors { fill: parent; margins: 5 }
                            spacing: 2

                            Repeater {
                                model: platBox.labels
                                Rectangle {
                                    width: parent.width; height: 30; radius: 5
                                    color: platBox.sel === index ? "#1e1e38"
                                         : (ph.containsMouse ? "#171730" : "transparent")
                                    HoverHandler { id: ph }
                                    Text {
                                        anchors { left: parent.left; leftMargin: 10; verticalCenter: parent.verticalCenter }
                                        text: modelData
                                        color: platBox.sel === index ? "#c4b5fd" : "#475569"
                                        font.pixelSize: 12
                                    }
                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: {
                                            platBox.sel = index
                                            bridge.setPlatform(platBox.opts[index])
                                            platBox.open = false
                                        }
                                        cursorShape: Qt.PointingHandCursor
                                    }
                                }
                            }
                        }
                    }
                }

                // ── Spacer ────────────────────────────────────────────
                Item { Layout.fillHeight: true }

                // ── Action buttons ────────────────────────────────────
                Rectangle {
                    Layout.fillWidth: true; height: 1; color: "#14142a"
                    Layout.bottomMargin: 20
                }

                // START ↔ STOP
                Rectangle {
                    id: startStopBtn
                    Layout.fillWidth: true; height: 42; radius: 9

                    property bool canStart: !bridge.pipelineRunning && !bridge.embeddingPending

                    gradient: Gradient {
                        orientation: Gradient.Horizontal
                        GradientStop {
                            position: 0.0
                            color: bridge.pipelineRunning ? "#7f1d1d"
                                 : (startStopBtn.canStart ? "#7c3aed" : "#181828")
                            Behavior on color { ColorAnimation { duration: 300 } }
                        }
                        GradientStop {
                            position: 1.0
                            color: bridge.pipelineRunning ? "#b91c1c"
                                 : (startStopBtn.canStart ? "#2563eb" : "#181828")
                            Behavior on color { ColorAnimation { duration: 300 } }
                        }
                    }

                    Row {
                        anchors.centerIn: parent; spacing: 8

                        Rectangle {
                            width: 6; height: 6
                            radius: bridge.pipelineRunning ? 1 : 3
                            color: "white"
                            opacity: (startStopBtn.canStart || bridge.pipelineRunning) ? 1.0 : 0.15
                            anchors.verticalCenter: parent.verticalCenter
                            Behavior on radius  { NumberAnimation { duration: 250 } }
                            Behavior on opacity { NumberAnimation { duration: 300 } }
                        }
                        Text {
                            text: bridge.pipelineRunning ? "STOP" : "START"
                            color: (startStopBtn.canStart || bridge.pipelineRunning) ? "white" : "#2a2a45"
                            font.pixelSize: 12; font.letterSpacing: 1.5; font.weight: Font.Medium
                            anchors.verticalCenter: parent.verticalCenter
                            Behavior on color { ColorAnimation { duration: 300 } }
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        enabled: startStopBtn.canStart || bridge.pipelineRunning
                        onClicked: bridge.pipelineRunning ? bridge.stopPipeline() : bridge.startPipeline()
                        cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                    }
                }

                // VIRTUAL CAM toggle
                Rectangle {
                    id: vcamToggleBtn
                    Layout.fillWidth: true; height: 36; radius: 8
                    Layout.topMargin: 8

                    property bool canVcam: bridge.pipelineRunning

                    color: bridge.virtualCamActive ? "#0a2218"
                         : (vh.containsMouse && canVcam ? "#1a1a2e" : "#12121e")
                    border.color: bridge.virtualCamActive ? "#10b981"
                                : (vh.containsMouse && canVcam ? "#2e2e50" : "#1e1e35")
                    border.width: 1
                    Behavior on color       { ColorAnimation { duration: 200 } }
                    Behavior on border.color { ColorAnimation { duration: 200 } }

                    Row {
                        anchors.centerIn: parent; spacing: 7

                        Rectangle {
                            width: 5; height: 5; radius: 2.5
                            color: bridge.virtualCamActive ? "#10b981" : "#2a2a45"
                            anchors.verticalCenter: parent.verticalCenter
                            Behavior on color { ColorAnimation { duration: 200 } }

                            SequentialAnimation on opacity {
                                running: bridge.virtualCamActive; loops: Animation.Infinite
                                NumberAnimation { to: 0.25; duration: 700; easing.type: Easing.InOutSine }
                                NumberAnimation { to: 1.0;  duration: 700; easing.type: Easing.InOutSine }
                            }
                        }
                        Text {
                            text: bridge.virtualCamActive ? "VCAM ON" : "VIRTUAL CAM"
                            color: bridge.virtualCamActive ? "#10b981"
                                 : vcamToggleBtn.canVcam  ? "#475569" : "#252545"
                            font.pixelSize: 12; font.letterSpacing: 1.5
                            anchors.verticalCenter: parent.verticalCenter
                            Behavior on color { ColorAnimation { duration: 200 } }
                        }
                    }

                    HoverHandler { id: vh }
                    MouseArea {
                        anchors.fill: parent
                        enabled: vcamToggleBtn.canVcam || bridge.virtualCamActive
                        onClicked: bridge.toggleVirtualCam()
                        cursorShape: (vcamToggleBtn.canVcam || bridge.virtualCamActive) ? Qt.PointingHandCursor : Qt.ArrowCursor
                    }
                }
            }
        }

        // ── Viewport (full area right of sidebar) ─────────────────────
        Rectangle {
            id: viewport
            anchors {
                top: parent.top; bottom: parent.bottom
                left: sidebar.right; right: parent.right
                margins: 14; leftMargin: 14
            }
            color: "#09090e"; radius: 14
            border.color: bridge.virtualCamActive ? "#10b981" : "#18182e"
            border.width: 1; clip: true
            Behavior on border.color { ColorAnimation { duration: 900 } }

            // Background feed: webcam when idle, processed when pipeline running
            FrameDisplay {
                id: bgFeed
                anchors.fill: parent
                source: bridge.pipelineRunning && bridge.liveVersion > 0 ? "live" : "webcam"
                frameVersion: bridge.pipelineRunning && bridge.liveVersion > 0
                              ? bridge.liveVersion : bridge.webcamVersion
                visible: bridge.webcamVersion > 0 || bridge.liveVersion > 0
            }

            // Placeholder when no feed
            Column {
                anchors.centerIn: parent; spacing: 12
                visible: bridge.webcamVersion === 0 && bridge.liveVersion === 0

                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: "◎"; color: "#1c1c35"; font.pixelSize: 48
                }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: "no camera feed"
                    color: "#252545"; font.pixelSize: 14
                }
            }

            // Virtual cam badge (bottom-left)
            Row {
                anchors { bottom: parent.bottom; left: parent.left; bottomMargin: 14; leftMargin: 16 }
                spacing: 7; visible: bridge.virtualCamActive

                Rectangle {
                    width: 5; height: 5; radius: 2.5; color: "#10b981"
                    anchors.verticalCenter: parent.verticalCenter

                    SequentialAnimation on opacity {
                        running: bridge.virtualCamActive; loops: Animation.Infinite
                        NumberAnimation { to: 0.25; duration: 900; easing.type: Easing.InOutSine }
                        NumberAnimation { to: 1.0;  duration: 900; easing.type: Easing.InOutSine }
                    }
                }
                Text {
                    text: "VCAM"
                    color: "#10b981"; font.pixelSize: 9
                    font.letterSpacing: 2.5; font.weight: Font.Medium
                    anchors.verticalCenter: parent.verticalCenter
                }
            }

            // ── Self-monitor PiP (top-right) ──────────────────────────
            Rectangle {
                id: miniScreen
                anchors { top: parent.top; right: parent.right; topMargin: 16; rightMargin: 16 }

                // 16:9 at ~22% of viewport width, min 200px
                width: Math.max(200, Math.round(viewport.width * 0.22))
                height: Math.round(width * 9 / 16)

                radius: 10
                color: "#111120"
                border.color: bridge.webcamVersion > 0 ? "#2e2e55" : "#1a1a30"
                border.width: 1
                clip: true

                property bool manuallyHidden: false
                visible: bridge.pipelineRunning && !manuallyHidden

                // Raw webcam feed (unprocessed self-view)
                FrameDisplay {
                    anchors.fill: parent
                    source: "webcam"
                    frameVersion: bridge.webcamVersion
                    visible: bridge.webcamVersion > 0
                }

                // Grey overlay when hidden or no feed
                Rectangle {
                    anchors.fill: parent; radius: parent.radius
                    color: "#111120"
                    visible: bridge.webcamVersion === 0
                }

                // "YOU" label — bottom-left
                Text {
                    anchors { bottom: parent.bottom; left: parent.left; bottomMargin: 7; leftMargin: 9 }
                    text: "YOU · UNPROCESSED"
                    color: bridge.webcamVersion > 0 ? "#33335a" : "#1e1e38"
                    font.pixelSize: 7; font.letterSpacing: 1.5
                }

                // Toggle button — top-right corner of PiP
                Rectangle {
                    id: pipToggle
                    anchors { top: parent.top; right: parent.right; topMargin: 7; rightMargin: 7 }
                    width: 22; height: 22; radius: 5
                    color: toggleHover.containsMouse ? "#1e1e38" : "transparent"
                    Behavior on color { ColorAnimation { duration: 100 } }

                    Text {
                        anchors.centerIn: parent
                        text: "✕"
                        color: toggleHover.containsMouse ? "#a78bfa" : "#33335a"
                        font.pixelSize: 10
                        Behavior on color { ColorAnimation { duration: 100 } }
                    }

                    HoverHandler { id: toggleHover }
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: miniScreen.manuallyHidden = true
                    }
                }
            }
        }
    }
}
