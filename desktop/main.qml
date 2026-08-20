import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import Phantom 1.0

Window {
    id: root
    visible: true
    width: 1600
    height: 620
    minimumWidth: 1080
    minimumHeight: 620
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

            // Separator dot
            Rectangle {
                width: 3; height: 3; radius: 1.5; color: "#1e1e38"
                anchors.verticalCenter: parent.verticalCenter
                visible: headerHint.text !== ""
            }

            // Hover hint label — shows tooltip text when any header button is hovered
            Text {
                id: headerHint
                property string hint: vcamHh.hovered  ? "Output swapped face to a virtual webcam for use in video calls"
                                    : enhHh.hovered   ? "Face restoration \u2014 off keeps the face softer, closer to the raw camera"
                                    : ""
                text: hint
                color: "#475569"; font.pixelSize: 11
                anchors.verticalCenter: parent.verticalCenter
                opacity: hint !== "" ? 1.0 : 0.0
                Behavior on opacity { NumberAnimation { duration: 150 } }
            }
        }

        Row {
            anchors { right: parent.right; rightMargin: 24; verticalCenter: parent.verticalCenter }
            spacing: 16

            // VCAM toggle (realtime mode only)
            Rectangle {
                visible: bridge.currentMode === "realtime"
                width: vcamRow.width + 20; height: 28; radius: 6
                anchors.verticalCenter: parent.verticalCenter
                color: bridge.virtualCamActive ? "#0a2218"
                     : vcamHh.containsMouse    ? "#151525" : "#0f0f1e"
                border.color: bridge.virtualCamActive ? "#10b981" : "#1e1e38"
                border.width: 1
                Behavior on color       { ColorAnimation { duration: 200 } }
                Behavior on border.color { ColorAnimation { duration: 200 } }

                Row {
                    id: vcamRow
                    anchors.centerIn: parent; spacing: 6

                    Rectangle {
                        width: 6; height: 6; radius: 3
                        color: bridge.virtualCamActive ? "#10b981" : "#333355"
                        anchors.verticalCenter: parent.verticalCenter
                        Behavior on color { ColorAnimation { duration: 200 } }

                        SequentialAnimation on opacity {
                            running: bridge.virtualCamActive; loops: Animation.Infinite
                            NumberAnimation { to: 0.25; duration: 700; easing.type: Easing.InOutSine }
                            NumberAnimation { to: 1.0;  duration: 700; easing.type: Easing.InOutSine }
                        }
                    }
                    Text {
                        text: "VCAM"
                        color: bridge.virtualCamActive ? "#10b981"
                             : vcamHh.containsMouse    ? "#64748b" : "#475569"
                        font.pixelSize: 10; font.letterSpacing: 1.5; font.weight: Font.Medium
                        anchors.verticalCenter: parent.verticalCenter
                        Behavior on color { ColorAnimation { duration: 200 } }
                    }
                }

                HoverHandler { id: vcamHh }
                MouseArea {
                    anchors.fill: parent
                    enabled: bridge.pipelineRunning || bridge.virtualCamActive
                    onClicked: bridge.toggleVirtualCam()
                    cursorShape: (bridge.pipelineRunning || bridge.virtualCamActive) ? Qt.PointingHandCursor : Qt.ArrowCursor
                }

            }

            // Enhance toggle (realtime mode only)
            Rectangle {
                visible: bridge.currentMode === "realtime"
                width: enhRow.width + 20; height: 28; radius: 6
                anchors.verticalCenter: parent.verticalCenter
                color: bridge.enhanceActive ? "#160a22"
                     : enhHh.containsMouse   ? "#151525" : "#0f0f1e"
                border.color: bridge.enhanceActive ? "#8b5cf6" : "#1e1e38"
                border.width: 1
                Behavior on color       { ColorAnimation { duration: 200 } }
                Behavior on border.color { ColorAnimation { duration: 200 } }

                Row {
                    id: enhRow
                    anchors.centerIn: parent; spacing: 6

                    Rectangle {
                        width: 6; height: 6; radius: 3
                        color: bridge.enhanceActive ? "#8b5cf6" : "#333355"
                        anchors.verticalCenter: parent.verticalCenter
                        Behavior on color { ColorAnimation { duration: 200 } }

                        SequentialAnimation on opacity {
                            running: bridge.enhanceActive; loops: Animation.Infinite
                            NumberAnimation { to: 0.25; duration: 700; easing.type: Easing.InOutSine }
                            NumberAnimation { to: 1.0;  duration: 700; easing.type: Easing.InOutSine }
                        }
                    }
                    Text {
                        text: "ENHANCE"
                        color: bridge.enhanceActive ? "#8b5cf6"
                             : enhHh.containsMouse  ? "#64748b" : "#475569"
                        font.pixelSize: 10; font.letterSpacing: 1.5; font.weight: Font.Medium
                        anchors.verticalCenter: parent.verticalCenter
                        Behavior on color { ColorAnimation { duration: 200 } }
                    }
                }

                HoverHandler { id: enhHh }
                MouseArea {
                    anchors.fill: parent
                    onClicked: bridge.toggleEnhance()
                    cursorShape: Qt.PointingHandCursor
                }

            }

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

            Rectangle { width: 1; height: 18; color: "#14142a"; anchors.verticalCenter: parent.verticalCenter }

            // ── Media tabs ────────────────────────────────────────
            // The top level of navigation: what kind of thing is being made.
            // The sidebar below picks the job within it.
            Rectangle {
                width: 132; height: 30; radius: 8
                anchors.verticalCenter: parent.verticalCenter
                color: "#0b0b16"
                border.color: "#1a1a30"; border.width: 1

                Row {
                    anchors { fill: parent; margins: 3 }
                    spacing: 2

                    Repeater {
                        model: [
                            { id: "video", label: "VIDEO" },
                            { id: "image", label: "IMAGE" },
                        ]

                        Rectangle {
                            width: (parent.width - 2) / 2; height: parent.height; radius: 6
                            property bool isActive: bridge.mediaTab === modelData.id
                            color: isActive ? "#1a1a30" : (mth.containsMouse ? "#111120" : "transparent")
                            border.color: isActive ? "#2e2e55" : "transparent"
                            border.width: 1
                            Behavior on color { ColorAnimation { duration: 120 } }

                            Text {
                                anchors.centerIn: parent
                                text: modelData.label
                                color: isActive ? "#c4b5fd" : "#334155"
                                font.pixelSize: 9; font.letterSpacing: 1.5; font.weight: Font.Medium
                                Behavior on color { ColorAnimation { duration: 120 } }
                            }

                            HoverHandler { id: mth }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: bridge.setMediaTab(modelData.id)
                                cursorShape: Qt.PointingHandCursor
                            }
                        }
                    }
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

                // ── Mode switcher ─────────────────────────────────────
                // The job within the selected media tab. Both tabs have two.
                Rectangle {
                    Layout.fillWidth: true; height: 34; radius: 8
                    color: "#0a0a14"
                    border.color: "#14142a"; border.width: 1
                    Layout.bottomMargin: 20

                    Row {
                        anchors { fill: parent; margins: 3 }
                        spacing: 2

                        Repeater {
                            // Video: "RENDER", not "BATCH" - batch reads as
                            // *many*, and this is one video processed offline
                            // rather than streamed.
                            //
                            // Image: the face is yours either way; what differs
                            // is whose picture it goes into. UPLOAD is one you
                            // bring, TEMPLATES is one we ship.
                            model: bridge.mediaTab === "video"
                                   ? [
                                       { id: "realtime", label: "LIVE"   },
                                       { id: "video",    label: "RENDER" },
                                     ]
                                   : [
                                       { id: "image",    label: "UPLOAD"    },
                                       { id: "template", label: "TEMPLATES" },
                                     ]

                            Rectangle {
                                width: (parent.width - 2) / 2; height: parent.height; radius: 6
                                property bool isActive: bridge.currentMode === modelData.id
                                color: isActive ? "#1a1a30" : (mh.containsMouse ? "#111120" : "transparent")
                                border.color: isActive ? "#2e2e55" : "transparent"
                                border.width: 1
                                Behavior on color { ColorAnimation { duration: 120 } }

                                Text {
                                    anchors.centerIn: parent
                                    text: modelData.label
                                    color: isActive ? "#c4b5fd" : "#334155"
                                    font.pixelSize: 9; font.letterSpacing: 1.5; font.weight: Font.Medium
                                    Behavior on color { ColorAnimation { duration: 120 } }
                                }

                                HoverHandler { id: mh }
                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: bridge.setMode(modelData.id)
                                    cursorShape: Qt.PointingHandCursor
                                }
                            }
                        }
                    }
                }

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

                // ══ REALTIME CONTROLS ══════════════════════════════════
                // visible only in LIVE mode
                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    visible: bridge.currentMode === "realtime"

                    ColumnLayout {
                        anchors { fill: parent }
                        spacing: 0

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

                        // ── Voice ─────────────────────────────────────────────
                        Text {
                            text: "VOICE"
                            color: "#252545"; font.pixelSize: 8; font.letterSpacing: 1.5
                            Layout.topMargin: 16; Layout.bottomMargin: 8
                        }

                        Rectangle {
                            id: voiceBox
                            Layout.fillWidth: true; height: 38; radius: 8
                            color: voiceHover.containsMouse ? "#1a1a2e" : "#12121e"
                            border.color: voiceBox.open ? "#3a3a60" : "#1e1e35"
                            border.width: 1
                            z: open ? 10 : 0
                            Behavior on color { ColorAnimation { duration: 130 } }

                            property var opts: ["none", "female", "male", "child", "deep"]
                            property var labels: ["None", "Female", "Male", "Child", "Deep"]
                            property int sel: 0
                            property bool open: false

                            Row {
                                anchors { fill: parent; leftMargin: 14; rightMargin: 10 }
                                Text {
                                    text: voiceBox.labels[voiceBox.sel]
                                    color: "#cbd5e1"; font.pixelSize: 12
                                    width: parent.width - 20
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                                Text { text: "⌄"; color: "#334155"; font.pixelSize: 10; anchors.verticalCenter: parent.verticalCenter }
                            }

                            HoverHandler { id: voiceHover }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: voiceBox.open = !voiceBox.open
                                cursorShape: Qt.PointingHandCursor
                            }

                            Rectangle {
                                visible: voiceBox.open
                                anchors.top: parent.bottom; anchors.topMargin: 4
                                anchors.left: parent.left
                                width: parent.width
                                height: voiceBox.opts.length * 32 + 10
                                radius: 8; color: "#12121e"
                                border.color: "#252545"; border.width: 1

                                Column {
                                    anchors { fill: parent; margins: 5 }
                                    spacing: 2

                                    Repeater {
                                        model: voiceBox.labels
                                        Rectangle {
                                            width: parent.width; height: 30; radius: 5
                                            color: voiceBox.sel === index ? "#1e1e38"
                                                 : (vh.containsMouse ? "#171730" : "transparent")
                                            HoverHandler { id: vh }
                                            Text {
                                                anchors { left: parent.left; leftMargin: 10; verticalCenter: parent.verticalCenter }
                                                text: modelData
                                                color: voiceBox.sel === index ? "#c4b5fd" : "#475569"
                                                font.pixelSize: 12
                                            }
                                            MouseArea {
                                                anchors.fill: parent
                                                onClicked: {
                                                    voiceBox.sel = index
                                                    bridge.setVoiceTemplate(voiceBox.opts[index])
                                                    voiceBox.open = false
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

                    }
                }

                // ══ BATCH CONTROLS ═════════════════════════════════════
                // visible in VIDEO and IMAGE modes
                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    visible: bridge.currentMode !== "realtime"

                    ColumnLayout {
                        anchors { fill: parent }
                        spacing: 0

                        // ── Target file ───────────────────────────────
                        Text {
                            text: bridge.currentMode === "video"
                                  ? "TARGET VIDEO"
                                  : bridge.currentMode === "template"
                                  ? "CHOOSE A SCENE"
                                  : "TARGET PHOTOS (MAX " + bridge.maxPhotoTargets + ")"
                            color: "#252545"; font.pixelSize: 8; font.letterSpacing: 1.5
                            Layout.bottomMargin: 8
                        }

                        // Select target button
                        Rectangle {
                            Layout.fillWidth: true; height: 38; radius: 8
                            // Templates are picked from the gallery below,
                            // not from the filesystem.
                            visible: !bridge.targetSet && bridge.currentMode !== "template"
                            color: tgtHover.containsMouse ? "#1a1a2e" : "#12121e"
                            border.color: tgtHover.containsMouse ? "#2e2e50" : "#1e1e35"
                            border.width: 1
                            Behavior on color      { ColorAnimation { duration: 130 } }
                            Behavior on border.color { ColorAnimation { duration: 130 } }

                            Row {
                                anchors.centerIn: parent; spacing: 8
                                Rectangle {
                                    width: 20; height: 20; radius: 10; color: "#0a1a35"
                                    anchors.verticalCenter: parent.verticalCenter
                                    Text { anchors.centerIn: parent; text: "+"; color: "#3b82f6"; font.pixelSize: 17 }
                                }
                                Text {
                                    text: bridge.currentMode === "video" ? "Select Video" : "Select Photos"
                                    color: "#60a5fa"; font.pixelSize: 12
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }

                            HoverHandler { id: tgtHover }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: bridge.selectTargetFile()
                                cursorShape: Qt.PointingHandCursor
                            }
                        }

                        // Target thumbnail card (video only — photo mode
                        // shows a tile per chosen image instead)
                        Rectangle {
                            Layout.fillWidth: true; height: 68; radius: 8
                            visible: bridge.targetSet && bridge.currentMode === "video"
                            color: "#12121e"
                            border.color: "#1a2a45"; border.width: 1
                            clip: true

                            // Image thumbnail (shown for image mode or if thumbnail available)
                            Image {
                                anchors.fill: parent
                                source: bridge.targetThumbnail !== ""
                                        ? "file:///" + bridge.targetThumbnail
                                        : ""
                                fillMode: Image.PreserveAspectCrop
                                smooth: true
                                visible: bridge.targetThumbnail !== ""
                            }

                            // Video icon placeholder
                            Column {
                                anchors.centerIn: parent; spacing: 4
                                visible: bridge.targetThumbnail === ""
                                Text { anchors.horizontalCenter: parent.horizontalCenter; text: "▶"; color: "#3b82f6"; font.pixelSize: 20 }
                            }

                            // bottom label
                            Rectangle {
                                anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
                                height: 24; color: "#d009090e"
                                Text {
                                    anchors { left: parent.left; leftMargin: 10; verticalCenter: parent.verticalCenter }
                                    text: bridge.targetLabel
                                    color: "#93c5fd"; font.pixelSize: 10
                                    elide: Text.ElideMiddle
                                    width: parent.width - 36
                                }
                            }

                            // × change button
                            Rectangle {
                                anchors { top: parent.top; right: parent.right; topMargin: 5; rightMargin: 5 }
                                width: 22; height: 22; radius: 5
                                color: tgtResetHover.containsMouse ? "#1d2c4e" : "#0a1428"
                                border.color: tgtResetHover.containsMouse ? "#2563eb" : "#1a2a45"
                                border.width: 1
                                Behavior on color { ColorAnimation { duration: 120 } }
                                Text { anchors.centerIn: parent; text: "×"; color: "#60a5fa"; font.pixelSize: 13 }
                                HoverHandler { id: tgtResetHover }
                                MouseArea {
                                    anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                    onClicked: bridge.selectTargetFile()
                                }
                            }
                        }

                        // ── Chosen photos ─────────────────────────────
                        // One tile per target, each carrying its own outcome:
                        // a job where two of four are skipped has no single
                        // verdict to show.
                        Flow {
                            Layout.fillWidth: true
                            visible: bridge.currentMode === "image" && bridge.photoTargets.length > 0
                            spacing: 6

                            Repeater {
                                model: bridge.photoTargets

                                Rectangle {
                                    width: 74; height: 74; radius: 8
                                    color: "#12121e"
                                    clip: true
                                    border.width: 1
                                    border.color: {
                                        var r = bridge.photoResults[index]
                                        if (!r) return "#1a2a45"
                                        return r.ok ? "#14532d" : "#4c1d24"
                                    }

                                    Image {
                                        anchors.fill: parent
                                        source: "file:///" + modelData
                                        fillMode: Image.PreserveAspectCrop
                                        smooth: true
                                        asynchronous: true
                                    }

                                    // Outcome badge — absent while pending
                                    Rectangle {
                                        anchors { top: parent.top; left: parent.left; topMargin: 4; leftMargin: 4 }
                                        width: 16; height: 16; radius: 8
                                        visible: bridge.photoResults[index] !== undefined
                                        color: {
                                            var r = bridge.photoResults[index]
                                            if (!r) return "transparent"
                                            return r.ok ? "#14532d" : "#4c1d24"
                                        }
                                        Text {
                                            anchors.centerIn: parent
                                            text: {
                                                var r = bridge.photoResults[index]
                                                if (!r) return ""
                                                return r.ok ? "\u2713" : "\u2715"
                                            }
                                            color: {
                                                var r = bridge.photoResults[index]
                                                if (!r) return "transparent"
                                                return r.ok ? "#86efac" : "#fca5a5"
                                            }
                                            font.pixelSize: 10
                                        }
                                    }

                                    // Skip reason, so a refusal says why
                                    Rectangle {
                                        anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
                                        height: 18
                                        visible: {
                                            var r = bridge.photoResults[index]
                                            return r !== undefined && !r.ok
                                        }
                                        color: "#d0090909"
                                        Text {
                                            anchors { fill: parent; leftMargin: 4; rightMargin: 4 }
                                            verticalAlignment: Text.AlignVCenter
                                            text: {
                                                var r = bridge.photoResults[index]
                                                return r ? r.reason : ""
                                            }
                                            color: "#fca5a5"; font.pixelSize: 7
                                            elide: Text.ElideRight
                                        }
                                    }

                                    // × remove, before the job runs
                                    Rectangle {
                                        anchors { top: parent.top; right: parent.right; topMargin: 4; rightMargin: 4 }
                                        width: 16; height: 16; radius: 4
                                        visible: !bridge.batchRunning
                                        color: rmHover.containsMouse ? "#1d2c4e" : "#c00a1428"
                                        Text { anchors.centerIn: parent; text: "×"; color: "#60a5fa"; font.pixelSize: 11 }
                                        HoverHandler { id: rmHover }
                                        MouseArea {
                                            anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                            onClicked: bridge.removePhotoTarget(index)
                                        }
                                    }
                                }
                            }
                        }

                        // ── Template gallery ──────────────────────────
                        // The scenes we ship. Selecting one sets it as the
                        // target; the source face is whatever was uploaded.
                        Flow {
                            Layout.fillWidth: true
                            visible: bridge.currentMode === "template"
                                     && bridge.templates.length > 0
                            spacing: 6

                            Repeater {
                                model: bridge.templates

                                Rectangle {
                                    width: 74; height: 74; radius: 8
                                    color: "#12121e"
                                    clip: true
                                    border.width: 1
                                    border.color: bridge.selectedTemplate === modelData.id
                                                  ? "#2e2e55" : "#1a2a45"

                                    Image {
                                        anchors.fill: parent
                                        source: modelData.thumbnail !== ""
                                                ? "file:///" + modelData.thumbnail
                                                : ""
                                        fillMode: Image.PreserveAspectCrop
                                        smooth: true
                                        asynchronous: true
                                        opacity: bridge.selectedTemplate === modelData.id ? 1.0 : 0.65
                                        Behavior on opacity { NumberAnimation { duration: 120 } }
                                    }

                                    // Name, so a scene without a thumbnail is
                                    // still identifiable
                                    Rectangle {
                                        anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
                                        height: 16; color: "#d009090e"
                                        Text {
                                            anchors { fill: parent; leftMargin: 4; rightMargin: 4 }
                                            verticalAlignment: Text.AlignVCenter
                                            text: modelData.name
                                            color: "#93c5fd"; font.pixelSize: 7
                                            elide: Text.ElideRight
                                        }
                                    }

                                    Rectangle {
                                        anchors { top: parent.top; left: parent.left; topMargin: 4; leftMargin: 4 }
                                        width: 16; height: 16; radius: 8
                                        visible: bridge.selectedTemplate === modelData.id
                                        color: "#1a1a30"
                                        Text {
                                            anchors.centerIn: parent
                                            text: "\u2713"; color: "#c4b5fd"; font.pixelSize: 10
                                        }
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        enabled: !bridge.batchRunning
                                        onClicked: bridge.selectTemplate(modelData.id)
                                        cursorShape: Qt.PointingHandCursor
                                    }
                                }
                            }
                        }

                        // Empty library, or one still loading
                        Rectangle {
                            Layout.fillWidth: true; height: 60; radius: 8
                            visible: bridge.currentMode === "template"
                                     && bridge.templates.length === 0
                            color: "#0d0d18"
                            border.color: "#14142a"; border.width: 1
                            Column {
                                anchors.centerIn: parent; spacing: 6
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: "no scenes available"
                                    color: "#334155"; font.pixelSize: 11
                                }
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: "retry"
                                    color: "#60a5fa"; font.pixelSize: 10
                                    MouseArea {
                                        anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                        onClicked: bridge.loadTemplates()
                                    }
                                }
                            }
                        }

                        // Add more, while under the cap
                        Rectangle {
                            Layout.fillWidth: true; height: 30; radius: 8
                            Layout.topMargin: 6
                            visible: bridge.currentMode === "image"
                                     && bridge.photoTargets.length > 0
                                     && bridge.photoTargets.length < bridge.maxPhotoTargets
                                     && !bridge.batchRunning
                            color: addHover.containsMouse ? "#1a1a2e" : "#12121e"
                            border.color: "#1e1e35"; border.width: 1
                            Text {
                                anchors.centerIn: parent
                                text: "Choose photos again"
                                color: "#60a5fa"; font.pixelSize: 10
                            }
                            HoverHandler { id: addHover }
                            MouseArea {
                                anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                onClicked: bridge.selectPhotoTargets()
                            }
                        }

                        // ── Output path ───────────────────────────────
                        // Photo mode derives one output per photo, beside the
                        // original, so there is nothing to choose here.
                        Text {
                            text: "OUTPUT PATH"
                            visible: bridge.currentMode === "video"
                            color: "#252545"; font.pixelSize: 8; font.letterSpacing: 1.5
                            Layout.topMargin: 16; Layout.bottomMargin: 8
                        }

                        Rectangle {
                            Layout.fillWidth: true; height: 38; radius: 8
                            visible: bridge.currentMode === "video"
                            color: outHover.containsMouse ? "#1a1a2e" : "#12121e"
                            border.color: bridge.outputPath !== "" ? "#1e2e1e" : "#1e1e35"
                            border.width: 1
                            Behavior on color { ColorAnimation { duration: 130 } }

                            Row {
                                anchors { fill: parent; leftMargin: 10; rightMargin: 10 }
                                spacing: 6
                                Text {
                                    text: bridge.outputPath !== "" ? bridge.outputPath.split("/").pop() : "auto"
                                    color: bridge.outputPath !== "" ? "#86efac" : "#334155"
                                    font.pixelSize: 11
                                    anchors.verticalCenter: parent.verticalCenter
                                    elide: Text.ElideLeft
                                    width: parent.width - 20
                                }
                                Text {
                                    text: "⌄"; color: "#334155"; font.pixelSize: 10
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }

                            HoverHandler { id: outHover }
                            MouseArea {
                                anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                onClicked: bridge.selectOutputPath()
                            }
                        }

                        // ── Spacer ────────────────────────────────────
                        Item { Layout.fillHeight: true }

                        // ── Action buttons ────────────────────────────
                        Rectangle {
                            Layout.fillWidth: true; height: 1; color: "#14142a"
                            Layout.bottomMargin: 20
                        }

                        // PROCESS ↔ STOP button
                        Rectangle {
                            id: batchBtn
                            Layout.fillWidth: true; height: 42; radius: 9

                            property bool canProcess: !bridge.batchRunning
                                                      && bridge.sourceSet
                                                      && bridge.targetSet
                                                      && !bridge.embeddingPending

                            gradient: Gradient {
                                orientation: Gradient.Horizontal
                                GradientStop {
                                    position: 0.0
                                    color: bridge.batchRunning ? "#7f1d1d"
                                         : (batchBtn.canProcess ? "#1d4ed8" : "#181828")
                                    Behavior on color { ColorAnimation { duration: 300 } }
                                }
                                GradientStop {
                                    position: 1.0
                                    color: bridge.batchRunning ? "#b91c1c"
                                         : (batchBtn.canProcess ? "#0ea5e9" : "#181828")
                                    Behavior on color { ColorAnimation { duration: 300 } }
                                }
                            }

                            Row {
                                anchors.centerIn: parent; spacing: 8

                                Rectangle {
                                    width: 6; height: 6
                                    radius: bridge.batchRunning ? 1 : 3
                                    color: "white"
                                    opacity: (batchBtn.canProcess || bridge.batchRunning) ? 1.0 : 0.15
                                    anchors.verticalCenter: parent.verticalCenter
                                    Behavior on radius  { NumberAnimation { duration: 250 } }
                                    Behavior on opacity { NumberAnimation { duration: 300 } }
                                }
                                Text {
                                    text: bridge.batchRunning ? "STOP" : "PROCESS"
                                    color: (batchBtn.canProcess || bridge.batchRunning) ? "white" : "#2a2a45"
                                    font.pixelSize: 12; font.letterSpacing: 1.5; font.weight: Font.Medium
                                    anchors.verticalCenter: parent.verticalCenter
                                    Behavior on color { ColorAnimation { duration: 300 } }
                                }
                            }

                            MouseArea {
                                anchors.fill: parent
                                enabled: batchBtn.canProcess || bridge.batchRunning
                                onClicked: bridge.batchRunning ? bridge.stopBatch() : bridge.startBatch()
                                cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                            }
                        }

                        // OPEN OUTPUT button (after complete)
                        Rectangle {
                            Layout.fillWidth: true; height: 36; radius: 8
                            Layout.topMargin: 8
                            visible: bridge.batchComplete
                            color: openHover.containsMouse ? "#0a2218" : "#081810"
                            border.color: openHover.containsMouse ? "#10b981" : "#0d3020"
                            border.width: 1
                            Behavior on color       { ColorAnimation { duration: 180 } }
                            Behavior on border.color { ColorAnimation { duration: 180 } }

                            Row {
                                anchors.centerIn: parent; spacing: 7
                                Text { text: "↗"; color: "#10b981"; font.pixelSize: 12; anchors.verticalCenter: parent.verticalCenter }
                                Text {
                                    text: "OPEN OUTPUT"
                                    color: "#10b981"
                                    font.pixelSize: 11; font.letterSpacing: 1.5
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }

                            HoverHandler { id: openHover }
                            MouseArea {
                                anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                onClicked: bridge.openOutputFolder()
                            }
                        }
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

            // ══ REALTIME VIEWPORT ══════════════════════════════════════
            Item {
                anchors.fill: parent
                visible: bridge.currentMode === "realtime"

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

                // ── Detection status badge (bottom-center) ────────────
                Rectangle {
                    anchors {
                        bottom: parent.bottom; horizontalCenter: parent.horizontalCenter
                        bottomMargin: 14
                    }
                    visible: bridge.pipelineRunning && bridge.detectionStatus !== ""
                    color: "#cc200a0a"
                    radius: 6
                    width: detectionLabel.width + 20
                    height: 22

                    Text {
                        id: detectionLabel
                        anchors.centerIn: parent
                        text: bridge.detectionStatus
                        color: "#f87171"
                        font.pixelSize: 10
                        font.letterSpacing: 1.5
                        font.weight: Font.Medium
                    }
                }

                // ── Model loading overlay ─────────────────────────────
                Rectangle {
                    anchors.fill: parent
                    visible: bridge.loadingMessage !== ""
                    color: "#d8090b12"

                    Column {
                        anchors.centerIn: parent
                        spacing: 18

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: bridge.loadingMessage
                            color: "#cbd5e1"
                            font.pixelSize: 13
                            font.letterSpacing: 0.8
                        }

                        Rectangle {
                            anchors.horizontalCenter: parent.horizontalCenter
                            width: 180; height: 2; radius: 1
                            color: "#1e1e38"
                            clip: true

                            Rectangle {
                                width: 80; height: parent.height; radius: parent.radius
                                gradient: Gradient {
                                    orientation: Gradient.Horizontal
                                    GradientStop { position: 0.0; color: "transparent" }
                                    GradientStop { position: 0.5; color: "#8b5cf6" }
                                    GradientStop { position: 1.0; color: "transparent" }
                                }

                                SequentialAnimation on x {
                                    running: bridge.loadingMessage !== ""
                                    loops: Animation.Infinite
                                    NumberAnimation { from: -80; to: 180; duration: 1400; easing.type: Easing.InOutSine }
                                }
                            }
                        }
                    }
                }

                // ── Self-monitor PiP (top-right) ──────────────────────
                Rectangle {
                    id: miniScreen
                    anchors { top: parent.top; right: parent.right; topMargin: 16; rightMargin: 16 }

                    width: Math.max(200, Math.round(viewport.width * 0.22))
                    height: Math.round(width * 9 / 16)

                    radius: 10
                    color: "#111120"
                    border.color: bridge.webcamVersion > 0 ? "#2e2e55" : "#1a1a30"
                    border.width: 1
                    clip: true

                    property bool manuallyHidden: false
                    visible: bridge.pipelineRunning && !manuallyHidden

                    FrameDisplay {
                        anchors.fill: parent
                        source: "webcam"
                        frameVersion: bridge.webcamVersion
                        visible: bridge.webcamVersion > 0
                    }

                    Rectangle {
                        anchors.fill: parent; radius: parent.radius
                        color: "#111120"
                        visible: bridge.webcamVersion === 0
                    }

                    Text {
                        anchors { bottom: parent.bottom; left: parent.left; bottomMargin: 7; leftMargin: 9 }
                        text: "YOU · UNPROCESSED"
                        color: bridge.webcamVersion > 0 ? "#33335a" : "#1e1e38"
                        font.pixelSize: 7; font.letterSpacing: 1.5
                    }

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

            // ══ BATCH VIEWPORT ═════════════════════════════════════════
            Item {
                anchors.fill: parent
                visible: bridge.currentMode !== "realtime"

                // Two-panel layout: target | result
                Row {
                    anchors { fill: parent; margins: 20 }
                    spacing: 16

                    // ── Target panel ──────────────────────────────────
                    Rectangle {
                        width: (parent.width - 16) / 2
                        height: parent.height
                        radius: 10
                        color: "#0d0d18"
                        border.color: "#14142a"; border.width: 1
                        clip: true

                        // Thumbnail (image mode or video poster)
                        Image {
                            anchors.fill: parent
                            source: bridge.targetThumbnail !== ""
                                    ? "file:///" + bridge.targetThumbnail
                                    : ""
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                            visible: bridge.targetThumbnail !== ""
                        }

                        // Placeholder when no target selected
                        Column {
                            anchors.centerIn: parent; spacing: 14
                            visible: !bridge.targetSet

                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                // A play glyph belongs to video; a still job
                                // gets a frame, and a template gets a gallery.
                                text: bridge.currentMode === "video" ? "▶"
                                    : bridge.currentMode === "template" ? "◳" : "▢"
                                color: "#1c1c35"; font.pixelSize: 42
                            }
                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                // Templates are not selected here — they are
                                // picked from the gallery in the sidebar, so
                                // this says where to look rather than implying
                                // something on this panel is clickable.
                                text: bridge.currentMode === "video" ? "select target video"
                                    : bridge.currentMode === "template" ? "choose a scene from the gallery"
                                    : "select target photos"
                                color: "#252545"; font.pixelSize: 13
                            }
                        }

                        // Video placeholder icon when target set but no thumbnail
                        Column {
                            anchors.centerIn: parent; spacing: 10
                            visible: bridge.targetSet && bridge.targetThumbnail === ""

                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: "▶"; color: "#1d4ed8"; font.pixelSize: 48
                            }
                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: bridge.targetLabel
                                color: "#60a5fa"; font.pixelSize: 12
                                elide: Text.ElideMiddle
                                width: 200
                                horizontalAlignment: Text.AlignHCenter
                            }
                        }

                        // Label badge
                        Rectangle {
                            anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
                            height: 28; color: "#c8090e14"
                            visible: bridge.targetSet
                            Text {
                                anchors { left: parent.left; leftMargin: 12; verticalCenter: parent.verticalCenter }
                                text: "TARGET"
                                color: "#334155"; font.pixelSize: 8; font.letterSpacing: 2
                            }
                        }
                    }

                    // ── Result panel ──────────────────────────────────
                    Rectangle {
                        width: (parent.width - 16) / 2
                        height: parent.height
                        radius: 10
                        color: "#0d0d18"
                        border.color: bridge.batchComplete ? "#0d2e1a" : "#14142a"
                        border.width: 1
                        clip: true
                        Behavior on border.color { ColorAnimation { duration: 600 } }

                        // Output image (video mode, after complete)
                        Image {
                            anchors.fill: parent
                            source: (bridge.batchComplete && bridge.currentMode === "video" && bridge.outputPath !== "")
                                    ? "file:///" + bridge.outputPath
                                    : ""
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                            visible: bridge.batchComplete && bridge.currentMode === "video" && bridge.outputPath !== ""
                            cache: false
                        }

                        // Swapped photos. A photo job has several outputs and
                        // may have skipped some, so the result is a grid with
                        // the skips named rather than one image.
                        Flow {
                            anchors { fill: parent; margins: 10 }
                            spacing: 6
                            // Templates report through the same per-photo
                            // results, being a photo job of one.
                            visible: (bridge.currentMode === "image"
                                      || bridge.currentMode === "template")
                                     && bridge.photoResults.length > 0

                            Repeater {
                                model: bridge.photoResults

                                Rectangle {
                                    width: (parent.width - 6) / 2
                                    height: (parent.height - 6) / 2
                                    radius: 8
                                    color: "#0a0a14"
                                    border.width: 1
                                    border.color: modelData.ok ? "#14532d" : "#4c1d24"
                                    clip: true

                                    Image {
                                        anchors.fill: parent
                                        source: modelData.output !== ""
                                                ? "file:///" + modelData.output
                                                : ""
                                        fillMode: Image.PreserveAspectFit
                                        smooth: true
                                        visible: modelData.output !== ""
                                        cache: false
                                    }

                                    // Why this one was skipped
                                    Column {
                                        anchors.centerIn: parent; spacing: 6
                                        width: parent.width - 20
                                        visible: !modelData.ok
                                        Text {
                                            width: parent.width
                                            horizontalAlignment: Text.AlignHCenter
                                            text: modelData.name
                                            color: "#64748b"; font.pixelSize: 10
                                            elide: Text.ElideMiddle
                                        }
                                        Text {
                                            width: parent.width
                                            horizontalAlignment: Text.AlignHCenter
                                            text: modelData.reason
                                            color: "#fca5a5"; font.pixelSize: 9
                                            wrapMode: Text.WordWrap
                                        }
                                    }
                                }
                            }
                        }

                        // Idle placeholder
                        Column {
                            anchors.centerIn: parent; spacing: 14
                            visible: !bridge.batchRunning && !bridge.batchComplete

                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: "◈"; color: "#1c1c35"; font.pixelSize: 42
                            }
                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: "result will appear here"
                                color: "#252545"; font.pixelSize: 13
                            }
                        }

                        // Processing overlay
                        Rectangle {
                            anchors.fill: parent; radius: parent.radius
                            color: "#d8090b12"
                            visible: bridge.batchRunning

                            Column {
                                anchors.centerIn: parent; spacing: 18

                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: "processing…"
                                    color: "#cbd5e1"; font.pixelSize: 13; font.letterSpacing: 0.8
                                }

                                Rectangle {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    width: 180; height: 2; radius: 1
                                    color: "#1e1e38"; clip: true

                                    Rectangle {
                                        width: 80; height: parent.height; radius: parent.radius
                                        gradient: Gradient {
                                            orientation: Gradient.Horizontal
                                            GradientStop { position: 0.0; color: "transparent" }
                                            GradientStop { position: 0.5; color: "#3b82f6" }
                                            GradientStop { position: 1.0; color: "transparent" }
                                        }

                                        SequentialAnimation on x {
                                            running: bridge.batchRunning; loops: Animation.Infinite
                                            NumberAnimation { from: -80; to: 180; duration: 1400; easing.type: Easing.InOutSine }
                                        }
                                    }
                                }
                            }
                        }

                        // Done overlay for video (no image to preview)
                        Rectangle {
                            anchors.fill: parent; radius: parent.radius
                            color: "#0c1a10"
                            visible: bridge.batchComplete && bridge.currentMode === "video"

                            Column {
                                anchors.centerIn: parent; spacing: 16

                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: "✓"; color: "#10b981"; font.pixelSize: 40
                                }
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: "processing complete"
                                    color: "#34d399"; font.pixelSize: 13; font.letterSpacing: 0.5
                                }
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: bridge.outputPath !== "" ? bridge.outputPath.split("/").pop() : ""
                                    color: "#6ee7b7"; font.pixelSize: 11
                                    elide: Text.ElideMiddle
                                    width: 220
                                    horizontalAlignment: Text.AlignHCenter
                                }
                            }
                        }

                        // Label badge
                        Rectangle {
                            anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
                            height: 28; color: "#c8090e14"
                            Text {
                                anchors { left: parent.left; leftMargin: 12; verticalCenter: parent.verticalCenter }
                                text: "OUTPUT"
                                color: bridge.batchComplete ? "#10b981" : "#334155"
                                font.pixelSize: 8; font.letterSpacing: 2
                                Behavior on color { ColorAnimation { duration: 400 } }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Auto-stop warning dialog ─────────────────────────────────────
    Rectangle {
        id: autoStopDialog
        visible: false
        anchors.fill: parent
        color: "#cc000000"
        z: 1000

        property int minutesLeft: 5

        Connections {
            target: bridge
            function onAutoStopWarning(minutes) {
                autoStopDialog.minutesLeft = minutes
                autoStopDialog.visible = true
                autoStopCountdown.restart()
            }
        }

        Timer {
            id: autoStopCountdown
            interval: 60000; repeat: true
            onTriggered: {
                autoStopDialog.minutesLeft -= 1
                if (autoStopDialog.minutesLeft <= 0) {
                    autoStopCountdown.stop()
                    autoStopDialog.visible = false
                }
            }
        }

        Rectangle {
            anchors.centerIn: parent
            width: 400; height: 180; radius: 12
            color: "#1a1a2e"
            border.color: "#ef4444"; border.width: 1

            Column {
                anchors.centerIn: parent; spacing: 16

                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: "AUTO-STOP WARNING"
                    color: "#ef4444"; font.pixelSize: 13
                    font.letterSpacing: 2; font.weight: Font.Medium
                }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: "Pod will stop in " + autoStopDialog.minutesLeft + " minute(s)"
                    color: "#e2e8f0"; font.pixelSize: 15
                }
                Row {
                    anchors.horizontalCenter: parent.horizontalCenter
                    spacing: 16

                    Rectangle {
                        width: extendText.width + 28; height: 34; radius: 6
                        color: extendMa.containsMouse ? "#1e3a5f" : "#172554"
                        border.color: "#3b82f6"; border.width: 1

                        Text {
                            id: extendText; anchors.centerIn: parent
                            text: "EXTEND"; color: "#3b82f6"
                            font.pixelSize: 11; font.letterSpacing: 1.5
                        }
                        MouseArea {
                            id: extendMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                bridge.keepAlive()
                                autoStopCountdown.stop()
                                autoStopDialog.visible = false
                            }
                        }
                    }

                    Rectangle {
                        width: dismissText.width + 28; height: 34; radius: 6
                        color: dismissMa.containsMouse ? "#1e1e30" : "#14142a"
                        border.color: "#334155"; border.width: 1

                        Text {
                            id: dismissText; anchors.centerIn: parent
                            text: "DISMISS"; color: "#64748b"
                            font.pixelSize: 11; font.letterSpacing: 1.5
                        }
                        MouseArea {
                            id: dismissMa; anchors.fill: parent
                            hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                autoStopCountdown.stop()
                                autoStopDialog.visible = false
                            }
                        }
                    }
                }
            }
        }
    }
}
