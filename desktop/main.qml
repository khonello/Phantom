import QtQuick
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Effects
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

    // ── Navigation geometry ───────────────────────────────────────────
    // Both levels of navigation are the same control: a two-up pane of
    // pills. The media tabs sit in the header, the mode tabs in the
    // sidebar. They read as a pair only if they are the same size, and the
    // header pane must never be the smaller of the two — it is the *outer*
    // level, and a top level that looks subordinate inverts the hierarchy.
    // One set of numbers, so the two cannot drift apart.
    readonly property int sidebarWidth: 256
    readonly property int sidebarPadding: 20
    readonly property int tabPaneWidth: sidebarWidth - sidebarPadding * 2
    readonly property int tabPaneHeight: 34

    // ── The app itself ────────────────────────────────────────────────
    // Everything except the overlays, in one item so it can be blurred as
    // one. The gate, the cards and the notice sit outside it — an overlay
    // that blurred itself would be unreadable.
    //
    // `layer.enabled` is toggled rather than `blurEnabled` so the effect
    // costs nothing at all while the notice is closed, which is almost
    // always: this is a live 30fps viewport underneath.
    Item {
        id: appBody
        anchors.fill: parent
        layer.enabled: bridge.faceNoticeOpen
        layer.effect: MultiEffect {
            blurEnabled: true
            blur: 1.0
            blurMax: 40
        }

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

                // A refusal used to render in the same grey as "idle", so the
                // one line telling someone their photo was rejected looked
                // like the app sitting there doing nothing.
                Text {
                    text: bridge.statusMessage
                    color: bridge.statusError ? "#f87171" : "#334155"
                    font.pixelSize: 12
                    anchors.verticalCenter: parent.verticalCenter
                    // Refusals name the file and say why, so they are long
                    // enough to shove the tabs off the header if left to grow.
                    width: Math.min(implicitWidth, Math.max(140, root.width - 620))
                    elide: Text.ElideRight
                    Behavior on color { ColorAnimation { duration: 160 } }
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

                // The one rule the app enforces everywhere, kept reachable
                // after the first-run card has been dismissed.
                Rectangle {
                    width: 20; height: 20; radius: 10
                    anchors.verticalCenter: parent.verticalCenter
                    color: noticeHint.containsMouse ? "#161628" : "transparent"
                    border.color: "#1e1e38"; border.width: 1
                    Behavior on color { ColorAnimation { duration: 120 } }

                    Text {
                        anchors.centerIn: parent
                        text: "?"
                        color: noticeHint.containsMouse ? "#c4b5fd" : "#334155"
                        font.pixelSize: 10; font.weight: Font.Medium
                        Behavior on color { ColorAnimation { duration: 120 } }
                    }

                    HoverHandler { id: noticeHint }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: bridge.openFaceNotice()
                        cursorShape: Qt.PointingHandCursor
                    }
                }

                // ── Media tabs ────────────────────────────────────────
                // The top level of navigation: what kind of thing is being made.
                // The sidebar below picks the job within it.
                Rectangle {
                    width: root.tabPaneWidth; height: root.tabPaneHeight; radius: 8
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
                width: root.sidebarWidth
                color: "#0d0d18"

                Rectangle {
                    anchors { top: parent.top; bottom: parent.bottom; right: parent.right }
                    width: 1; color: "#14142a"
                }

                ColumnLayout {
                    anchors { fill: parent; margins: root.sidebarPadding
                              bottomMargin: root.sidebarPadding }
                    spacing: 0

                    // ── Mode switcher ─────────────────────────────────────
                    // The job within the selected media tab. Both tabs have two.
                    Rectangle {
                        Layout.fillWidth: true; height: root.tabPaneHeight; radius: 8
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

                                        // Needs a face chosen. Said before the
                                        // job runs, not after: this is the one
                                        // refusal the operator can still fix.
                                        Rectangle {
                                            anchors { bottom: parent.bottom; left: parent.left; right: parent.right }
                                            height: 18
                                            visible: bridge.photoNeedsFace[index] === true
                                                     && bridge.photoResults[index] === undefined
                                            color: "#d0180f02"
                                            Text {
                                                anchors { fill: parent; leftMargin: 4; rightMargin: 4 }
                                                verticalAlignment: Text.AlignVCenter
                                                text: "choose a face"
                                                color: "#fbbf24"; font.pixelSize: 7
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
                // A frame edge, not a status light. This outline used to turn
                // green with the virtual camera, which only ever showed while the
                // viewport was empty: FrameDisplay aspect-crops to fill and is
                // anchored to the full rect, so the first frame paints straight
                // over the border — and clip is a rectangular scissor, so it
                // squares off the radius too. A signal visible only when there is
                // nothing to look at is worse than no signal, and the camera is
                // already reported twice: the header pill and the corner badge.
                border.color: "#18182e"
                border.width: 1; clip: true

                // ══ REALTIME VIEWPORT ══════════════════════════════════════
                Item {
                    anchors {
                        fill: parent
                        bottomMargin: filterStrip.reserved
                        rightMargin: effectRail.reserved
                    }
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

                    // ── Guard badge (above the detection badge) ───────────
                    // Why the picture has stopped moving. The pipeline holds
                    // the last swapped frame when it will not swap, which is
                    // deliberately indistinguishable from a network hiccup to
                    // everyone else on the call — the operator is the one
                    // person who has to know it was a guard.
                    //
                    // Here rather than on the frame: the frame reaches every
                    // participant, so anything drawn onto it announces the
                    // failure to all of them.
                    Rectangle {
                        anchors {
                            bottom: parent.bottom; horizontalCenter: parent.horizontalCenter
                            bottomMargin: 42
                        }
                        visible: bridge.pipelineRunning && bridge.guardReason !== ""
                        color: "#cc2a1c05"
                        radius: 6
                        width: Math.min(guardLabel.implicitWidth + 20, parent.width - 40)
                        height: 22

                        Text {
                            id: guardLabel
                            anchors.centerIn: parent
                            width: parent.width - 20
                            text: "HOLDING · " + bridge.guardReason
                            elide: Text.ElideRight
                            horizontalAlignment: Text.AlignHCenter
                            color: "#fbbf24"
                            font.pixelSize: 10
                            font.letterSpacing: 1.5
                            font.weight: Font.Medium
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
                    anchors {
                        fill: parent
                        bottomMargin: filterStrip.reserved
                        rightMargin: effectRail.reserved
                    }
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

                // ══ FACE PICKER ════════════════════════════════════════════
                // Which face in this photo. Over the whole viewport, filter
                // panels included, because the question has to be answered
                // before anything else in this mode means much.
                //
                // Here rather than on the 74px sidebar tile: that tile crops
                // to fill, so a point clicked on it cannot be mapped back to a
                // point in the photo. This one fits the photo whole, which is
                // what makes `paintedWidth`/`paintedHeight` the exact rect the
                // normalised boxes belong in.
                Rectangle {
                    anchors.fill: parent
                    radius: parent.radius
                    color: "#f2090b12"
                    visible: bridge.pickerOpen
                    z: 50

                    Text {
                        id: pickerTitle
                        anchors { top: parent.top; topMargin: 18; horizontalCenter: parent.horizontalCenter }
                        text: bridge.pickerTotal > 1
                              ? "WHICH FACE?  ·  PHOTO " + bridge.pickerPosition + " OF " + bridge.pickerTotal
                              : "WHICH FACE?"
                        color: "#c4b5fd"
                        font.pixelSize: 10; font.letterSpacing: 2.5; font.weight: Font.Medium
                    }

                    Text {
                        id: pickerHint
                        anchors { top: pickerTitle.bottom; topMargin: 6; horizontalCenter: parent.horizontalCenter }
                        text: "this photo has more than one — click the one to swap"
                        color: "#475569"
                        font.pixelSize: 11
                    }

                    Item {
                        id: pickerStage
                        anchors {
                            top: pickerHint.bottom; bottom: pickerCancel.top
                            left: parent.left; right: parent.right
                            topMargin: 14; bottomMargin: 14
                            leftMargin: 24; rightMargin: 24
                        }

                        Image {
                            id: pickerImage
                            anchors.fill: parent
                            source: bridge.pickerPhoto === "" ? "" : "file:///" + bridge.pickerPhoto
                            fillMode: Image.PreserveAspectFit
                            asynchronous: true
                            cache: false
                        }

                        // The photo's own rect inside the letterboxed item.
                        // Every box is normalised against the photo, not
                        // against the space it happens to be shown in.
                        Item {
                            id: pickerFrame
                            x: (pickerStage.width - pickerImage.paintedWidth) / 2
                            y: (pickerStage.height - pickerImage.paintedHeight) / 2
                            width: pickerImage.paintedWidth
                            height: pickerImage.paintedHeight

                            Repeater {
                                model: bridge.pickerBoxes

                                Rectangle {
                                    x: modelData.x * pickerFrame.width
                                    y: modelData.y * pickerFrame.height
                                    width: modelData.w * pickerFrame.width
                                    height: modelData.h * pickerFrame.height
                                    color: boxHover.containsMouse ? "#338b5cf6" : "transparent"
                                    border.color: boxHover.containsMouse ? "#c4b5fd" : "#8b5cf6"
                                    border.width: 2
                                    radius: 4
                                    Behavior on color { ColorAnimation { duration: 100 } }

                                    HoverHandler { id: boxHover }
                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: bridge.chooseFace(index)
                                        cursorShape: Qt.PointingHandCursor
                                    }
                                }
                            }
                        }
                    }

                    Rectangle {
                        id: pickerCancel
                        anchors { bottom: parent.bottom; bottomMargin: 18; horizontalCenter: parent.horizontalCenter }
                        width: 100; height: 28; radius: 8
                        color: cancelHover.containsMouse ? "#1a1a30" : "transparent"
                        border.color: "#1e1e38"; border.width: 1
                        Behavior on color { ColorAnimation { duration: 120 } }

                        Text {
                            anchors.centerIn: parent
                            text: "CANCEL"
                            color: "#475569"
                            font.pixelSize: 9; font.letterSpacing: 2; font.weight: Font.Medium
                        }

                        HoverHandler { id: cancelHover }
                        MouseArea {
                            anchors.fill: parent
                            onClicked: bridge.cancelPicker()
                            cursorShape: Qt.PointingHandCursor
                        }
                    }
                }

                // ══ FILTER STRIP ═══════════════════════════════════════════
                // The filter controls sit *under* the body rather than replacing
                // it, so the same thing is on screen in every mode and a look is
                // judged against whatever that mode was already showing. A
                // separate window had to invent a preview of its own, and got it
                // wrong the moment the image tab was open — it showed the live
                // camera in a mode that has no live camera.
                Item {
                    id: filterStrip
                    // What the body gives up. One number, read by both viewports,
                    // so the body and the strip cannot disagree about the split.
                    property int reserved: bridge.filterPanel ? height + 12 : 0

                    anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
                    height: 46
                    visible: bridge.filterPanel

                    Row {
                        anchors { left: parent.left; leftMargin: 20; bottom: parent.bottom; bottomMargin: 16 }
                        spacing: 6

                        Repeater {
                            model: bridge.filterList

                            Rectangle {
                                width: 76; height: 30; radius: 6
                                property bool isActive: bridge.activeFilter === modelData.key
                                color: isActive ? "#1a1a30" : (fh.containsMouse ? "#111120" : "#0d0d18")
                                border.color: isActive ? "#2e2e55" : "#14142a"
                                border.width: 1
                                Behavior on color { ColorAnimation { duration: 120 } }

                                Text {
                                    anchors.centerIn: parent
                                    text: modelData.name
                                    color: isActive ? "#c4b5fd" : "#475569"
                                    font.pixelSize: 10; font.letterSpacing: 0.8
                                }

                                HoverHandler { id: fh }
                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: bridge.selectFilter(modelData.key)
                                    cursorShape: Qt.PointingHandCursor
                                }
                            }
                        }

                    }

                    // Choosing is not applying: a look can be picked and seen here
                    // without it reaching a call until this is pressed. Sits beside
                    // the show/hide control rather than at the end of the chips,
                    // because it acts on the whole panel and not on any one chip.
                    Rectangle {
                        anchors { right: parent.right; rightMargin: 124; bottom: parent.bottom; bottomMargin: 16 }
                        width: 76; height: 30; radius: 6
                        color: bridge.filtersEnabled ? "#0d2e1a" : "#1a1a30"
                        border.color: bridge.filtersEnabled ? "#14532d" : "#2e2e55"
                        border.width: 1
                        opacity: applyArea.enabled ? 1.0 : 0.3
                        Behavior on color { ColorAnimation { duration: 160 } }

                        Text {
                            anchors.centerIn: parent
                            text: bridge.filtersEnabled ? "APPLIED" : "APPLY"
                            color: bridge.filtersEnabled ? "#86efac" : "#c4b5fd"
                            font.pixelSize: 9; font.letterSpacing: 1.0
                        }

                        MouseArea {
                            id: applyArea
                            anchors.fill: parent
                            enabled: bridge.activeFilter !== "none"
                                     || bridge.activeEffect !== "none"
                                     || bridge.filtersEnabled
                            onClicked: bridge.toggleFilters()
                            cursorShape: Qt.PointingHandCursor
                        }
                    }
                }

                // ══ EFFECT RAIL ════════════════════════════════════════════
                // The overlays — confetti and the rest. Vertical on the right,
                // because they are a different kind of thing from a grade: one
                // changes the colour of the picture, the other puts something on
                // top of it. Shown and hidden by the same control as the strip,
                // since they are two halves of one panel.
                Item {
                    id: effectRail
                    property int reserved: bridge.filterPanel ? width + 12 : 0

                    anchors {
                        top: parent.top; right: parent.right
                        bottom: parent.bottom; bottomMargin: filterStrip.height
                    }
                    width: 96
                    visible: bridge.filterPanel

                    Column {
                        anchors { top: parent.top; topMargin: 20; horizontalCenter: parent.horizontalCenter }
                        spacing: 6

                        Repeater {
                            model: bridge.effectList

                            Rectangle {
                                width: 76; height: 30; radius: 6
                                property bool isActive: bridge.activeEffect === modelData.key
                                color: isActive ? "#1a1a30" : (eh.containsMouse ? "#111120" : "#0d0d18")
                                border.color: isActive ? "#2e2e55" : "#14142a"
                                border.width: 1
                                Behavior on color { ColorAnimation { duration: 120 } }

                                Text {
                                    anchors.centerIn: parent
                                    text: modelData.name
                                    color: isActive ? "#c4b5fd" : "#475569"
                                    font.pixelSize: 10; font.letterSpacing: 0.8
                                }

                                HoverHandler { id: eh }
                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: bridge.selectEffect(modelData.key)
                                    cursorShape: Qt.PointingHandCursor
                                }
                            }
                        }
                    }
                }

                // ══ FILTER TOGGLE ══════════════════════════════════════════
                // Same shape as every other control here — a rounded rectangle,
                // not a pill.
                Rectangle {
                    anchors { right: parent.right; bottom: parent.bottom; margins: 16 }
                    width: 96; height: 30; radius: 6
                    color: panelHover.containsMouse ? "#1a1a30" : "#0d0d18"
                    border.color: bridge.filtersEnabled ? "#2e2e55" : "#14142a"
                    border.width: 1
                    Behavior on color { ColorAnimation { duration: 130 } }

                    Row {
                        anchors.centerIn: parent; spacing: 6

                        Rectangle {
                            width: 5; height: 5; radius: 2.5
                            anchors.verticalCenter: parent.verticalCenter
                            color: bridge.filtersEnabled ? "#a78bfa" : "#333355"
                            Behavior on color { ColorAnimation { duration: 200 } }
                        }
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            text: bridge.filterPanel ? "HIDE" : "FILTERS"
                            color: panelHover.containsMouse ? "#c4b5fd" : "#64748b"
                            font.pixelSize: 9; font.letterSpacing: 1.4; font.weight: Font.Medium
                            Behavior on color { ColorAnimation { duration: 130 } }
                        }
                    }

                    HoverHandler { id: panelHover }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: bridge.toggleFilterPanel()
                        cursorShape: Qt.PointingHandCursor
                    }
                }

            }
        }

    }

    // ── One-face notice ──────────────────────────────────────────────
    // Shown once, on a first run. Outside `appBody` so the blur applies to
    // the app behind it and not to itself.
    //
    // It says three rules rather than one, because "exactly one face" stopped
    // being true when the picker landed: a target photo may hold several, so
    // long as the operator says which. A rule stated more strictly than the
    // app enforces it teaches people to distrust the next one.
    Rectangle {
        id: faceNotice
        visible: bridge.faceNoticeOpen
        anchors.fill: parent
        color: "#cc000000"
        z: 1800

        // Dismissing by clicking away is the point of a scrim; the MouseArea
        // also swallows clicks that would otherwise land on the app behind.
        MouseArea {
            anchors.fill: parent
            hoverEnabled: true
            onClicked: bridge.dismissFaceNotice()
        }

        Rectangle {
            anchors.centerIn: parent
            width: 460
            height: noticeColumn.implicitHeight + 56
            radius: 12
            color: "#12121f"
            border.color: "#2e2e55"; border.width: 1

            // Clicks on the card are not clicks outside it.
            MouseArea { anchors.fill: parent; hoverEnabled: true }

            Column {
                id: noticeColumn
                anchors {
                    left: parent.left; right: parent.right
                    verticalCenter: parent.verticalCenter
                    leftMargin: 28; rightMargin: 28
                }
                spacing: 14

                Text {
                    text: "ONE FACE"
                    color: "#c4b5fd"
                    font.pixelSize: 11; font.letterSpacing: 3; font.weight: Font.Medium
                }

                Text {
                    width: parent.width
                    wrapMode: Text.WordWrap
                    text: "A swap replaces one person with one other person. "
                          + "Where there is any doubt about which, Phantom stops "
                          + "rather than guessing — a swap of the wrong face looks "
                          + "like it worked."
                    color: "#94a3b8"
                    font.pixelSize: 12; lineHeight: 1.35
                }

                Repeater {
                    model: [
                        { k: "Your face photos",
                          v: "Exactly one face, every time. A photo with two "
                             + "people in it is refused." },
                        { k: "Live and video",
                          v: "One face in shot. A second face pauses the swap, "
                             + "and stops a render." },
                        { k: "Target photos",
                          v: "More than one is fine — you will be asked which "
                             + "one to swap." },
                    ]

                    Row {
                        width: noticeColumn.width
                        spacing: 12

                        Rectangle {
                            width: 3; height: ruleColumn.implicitHeight
                            color: "#8b5cf6"; radius: 1.5
                        }

                        Column {
                            id: ruleColumn
                            width: noticeColumn.width - 15
                            spacing: 3

                            Text {
                                text: modelData.k
                                color: "#e2e8f0"
                                font.pixelSize: 11; font.weight: Font.Medium
                            }
                            Text {
                                width: parent.width
                                wrapMode: Text.WordWrap
                                text: modelData.v
                                color: "#64748b"
                                font.pixelSize: 11; lineHeight: 1.3
                            }
                        }
                    }
                }

                Rectangle {
                    width: 120; height: 32; radius: 8
                    color: noticeOk.containsMouse ? "#2e2e55" : "#1a1a30"
                    border.color: "#2e2e55"; border.width: 1
                    Behavior on color { ColorAnimation { duration: 120 } }

                    Text {
                        anchors.centerIn: parent
                        text: "GOT IT"
                        color: "#c4b5fd"
                        font.pixelSize: 10; font.letterSpacing: 2; font.weight: Font.Medium
                    }

                    HoverHandler { id: noticeOk }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: bridge.dismissFaceNotice()
                        cursorShape: Qt.PointingHandCursor
                    }
                }
            }
        }
    }

    // ── Access gate ──────────────────────────────────────────────────
    // Covers everything when the machine has no time left. Above the
    // auto-stop dialog in z-order because an expired session should not be
    // showing a countdown for a pod it can no longer reach.
    //
    // Deliberately opaque rather than translucent: this is not a modal over
    // the app, it is the app not being available yet. A dimmed but visible
    // interface behind it reads as something to click past.
    Rectangle {
        id: authGate
        visible: bridge.authRequired
        anchors.fill: parent
        color: "#09090e"
        z: 2000

        // QML's font.family takes one family name, not a CSS-style list — a
        // comma-separated string is treated as a single (missing) family and
        // silently falls back to the default, losing the fixed pitch that
        // makes a grouped code readable. Pick per platform instead.
        readonly property string monoFamily:
              Qt.platform.os === "windows" ? "Consolas"
            : Qt.platform.os === "osx"     ? "Menlo"
            :                               "Monospace"

        // Swallow every click and key press that reaches the backdrop, so
        // nothing behind the gate can be operated through it.
        MouseArea { anchors.fill: parent; hoverEnabled: true }

        Column {
            anchors.centerIn: parent
            spacing: 22
            width: 360

            Row {
                anchors.horizontalCenter: parent.horizontalCenter
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

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "Enter your access code"
                color: "#94a3b8"; font.pixelSize: 13
            }

            // The field. Uppercases as you type and accepts the grouped form,
            // so a code read off a phone screen can be typed exactly as seen.
            Rectangle {
                id: codeBox
                anchors.horizontalCenter: parent.horizontalCenter
                width: 280; height: 46; radius: 8
                color: "#0f0f18"
                border.width: 1
                border.color: bridge.authError !== "" ? "#ef4444"
                            : codeField.activeFocus  ? "#3b82f6"
                            : "#1e1e38"

                TextInput {
                    id: codeField
                    anchors.fill: parent
                    anchors.margins: 12
                    verticalAlignment: TextInput.AlignVCenter
                    horizontalAlignment: TextInput.AlignHCenter
                    color: "#e2e8f0"
                    font.pixelSize: 18
                    font.letterSpacing: 3
                    font.family: authGate.monoFamily
                    enabled: !bridge.authChecking
                    focus: authGate.visible
                    // 10 characters plus the separator.
                    maximumLength: 11
                    onTextChanged: {
                        var up = text.toUpperCase()
                        if (up !== text) text = up
                    }
                    onAccepted: authGate.submit()

                    Text {
                        anchors.centerIn: parent
                        visible: codeField.text === ""
                        text: "XXXXX-XXXXX"
                        color: "#2a2a44"
                        font: codeField.font
                    }
                }
            }

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                width: parent.width
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
                text: bridge.authError
                color: "#ef4444"; font.pixelSize: 12
                visible: bridge.authError !== "" && !bridge.authChecking
            }

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "Checking…"
                color: "#64748b"; font.pixelSize: 12
                visible: bridge.authChecking
            }

            Rectangle {
                anchors.horizontalCenter: parent.horizontalCenter
                width: 280; height: 40; radius: 8
                opacity: (bridge.authChecking || codeField.text.length === 0) ? 0.4 : 1.0
                color: unlockMa.containsMouse ? "#1e3a5f" : "#172554"
                border.color: "#3b82f6"; border.width: 1

                Text {
                    anchors.centerIn: parent
                    text: "START SESSION"
                    color: "#3b82f6"
                    font.pixelSize: 11; font.letterSpacing: 1.5
                }
                MouseArea {
                    id: unlockMa
                    anchors.fill: parent
                    hoverEnabled: true
                    enabled: !bridge.authChecking && codeField.text.length > 0
                    cursorShape: Qt.PointingHandCursor
                    onClicked: authGate.submit()
                }
            }

            // Reaching the server is our problem, not the customer's, so it
            // gets its own affordance rather than being folded into the error
            // text above the code field.
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "Retry connection"
                color: retryMa.containsMouse ? "#94a3b8" : "#475569"
                font.pixelSize: 11
                font.underline: retryMa.containsMouse
                visible: bridge.authError.indexOf("licence server") !== -1

                MouseArea {
                    id: retryMa
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: bridge.checkAuth()
                }
            }
        }

        function submit() {
            if (bridge.authChecking || codeField.text.length === 0)
                return
            bridge.submitCode(codeField.text)
        }

        // Clear the field once the gate opens, so a code is never left sitting
        // on screen after it has been spent.
        onVisibleChanged: if (!visible) codeField.text = ""
    }

    // ── Session countdown ────────────────────────────────────────────
    // Only appears in the last ten minutes. A permanent clock on a paid hour
    // invites watching it instead of using it.
    Rectangle {
        visible: !bridge.authRequired && bridge.authMinutes > 0 && bridge.authMinutes <= 10
        anchors { top: parent.top; right: parent.right; topMargin: 14; rightMargin: 20 }
        width: sessionLeft.width + 20; height: 24; radius: 4
        color: "#1a1020"
        border.color: "#ef4444"; border.width: 1
        z: 900

        Text {
            id: sessionLeft
            anchors.centerIn: parent
            text: bridge.authMinutes + " MIN LEFT"
            color: "#ef4444"; font.pixelSize: 10; font.letterSpacing: 1.5
        }
    }

    // ── Session ended ────────────────────────────────────────────────
    // An anchored card, not the full gate and not anything drawn on the
    // picture. Two reasons:
    //
    //   The operator may still be in a call. Covering the whole window at the
    //   moment their time runs out hides the app from someone who needs to see
    //   it; the frozen preview underneath is information, not decoration.
    //
    //   Nothing is ever composited onto the frame. What the call sees is the
    //   last swapped frame, unchanged, still being sent — the notice lives in
    //   this window and travels nowhere.
    Rectangle {
        id: sessionEndedCard
        visible: bridge.sessionExpired
        anchors.fill: parent
        color: "#99000000"
        z: 1500

        MouseArea { anchors.fill: parent; hoverEnabled: true }

        Rectangle {
            anchors.centerIn: parent
            width: 420; height: 210; radius: 12
            color: "#12121f"
            border.color: "#8b5cf6"; border.width: 1

            Column {
                anchors.centerIn: parent
                spacing: 14
                width: parent.width - 48

                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: "SESSION ENDED"
                    color: "#8b5cf6"; font.pixelSize: 12
                    font.letterSpacing: 2.5; font.weight: Font.Medium
                }
                Text {
                    width: parent.width
                    horizontalAlignment: Text.AlignHCenter
                    wrapMode: Text.WordWrap
                    text: bridge.sessionReason
                    color: "#e2e8f0"; font.pixelSize: 14
                }
                Text {
                    width: parent.width
                    horizontalAlignment: Text.AlignHCenter
                    wrapMode: Text.WordWrap
                    text: "Your camera is still showing the last frame. It will "
                        + "keep doing so until you close Phantom."
                    color: "#64748b"; font.pixelSize: 11; lineHeight: 1.3
                }

                Rectangle {
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: 200; height: 38; radius: 8
                    color: newCodeMa.containsMouse ? "#1e3a5f" : "#172554"
                    border.color: "#3b82f6"; border.width: 1

                    Text {
                        anchors.centerIn: parent
                        text: "ENTER A NEW CODE"
                        color: "#3b82f6"
                        font.pixelSize: 11; font.letterSpacing: 1.5
                    }
                    MouseArea {
                        id: newCodeMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: bridge.enterNewCode()
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
