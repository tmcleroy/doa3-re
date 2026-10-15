// SPDX-License-Identifier: GPL-3.0-or-later
// macOS launcher: pick resolution, anti-aliasing, volume and widescreen,
// remember them in private/launcher.json, then exit 0 so launcher.sh starts
// the game. Mirrors tools/launcher.ps1; SMAA needs the D3D11 presenter.
// Exits 1 if the window is closed.
import AppKit

let root = URL(fileURLWithPath: CommandLine.arguments.count > 1
    ? CommandLine.arguments[1] : FileManager.default.currentDirectoryPath)
let settingsURL = root.appendingPathComponent("private/launcher.json")

let heights = [480, 720, 1080, 1440, 2160]
let samples = [1, 2, 4, 8]
let volumes = [100, 75, 50, 25, 0]
// Converted DOAX characters (private/doax-<name>.dxc), for the experimental
// costume injection; "Stock" keeps the fighter's own model.
let costumes: [String] = {
    let dir = root.appendingPathComponent("private")
    let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
    return names.filter { $0.hasPrefix("doax-") && $0.hasSuffix(".dxc") }
        .map { String($0.dropFirst(5).dropLast(4)) }.sorted()
}()

var settings: [String: Any] = [:]
if let data = try? Data(contentsOf: settingsURL),
   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
    settings = json
}
func intSetting(_ key: String, _ fallback: Int) -> Int {
    (settings[key] as? NSNumber)?.intValue ?? fallback
}
func boolSetting(_ key: String) -> Bool {
    (settings[key] as? NSNumber)?.boolValue ?? false
}
let screenHeight: Int = {
    guard let screen = NSScreen.screens.first ?? NSScreen.main else { return 1080 }
    return Int(screen.frame.height * screen.backingScaleFactor)
}()

final class Launcher: NSObject, NSApplicationDelegate, NSWindowDelegate {
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 260, height: costumes.isEmpty ? 214 : 278),
                          styleMask: [.titled, .closable], backing: .buffered, defer: false)
    let resolution = NSPopUpButton()
    let msaa = NSPopUpButton()
    let volume = NSPopUpButton()
    let costumeP1 = NSPopUpButton()
    let costumeP2 = NSPopUpButton()
    let smaa = NSButton(checkboxWithTitle: "SMAA", target: nil, action: nil)
    let widescreen = NSButton(checkboxWithTitle: "Widescreen (16:9)", target: nil, action: nil)
    var launched = false

    func addRow(_ title: String, _ popup: NSPopUpButton, _ items: [String], _ index: Int, _ y: CGFloat) {
        let label = NSTextField(labelWithString: title)
        label.frame = NSRect(x: 16, y: y + 3, width: 84, height: 20)
        popup.frame = NSRect(x: 104, y: y, width: 140, height: 26)
        popup.addItems(withTitles: items)
        popup.selectItem(at: max(0, index))
        window.contentView!.addSubview(label)
        window.contentView!.addSubview(popup)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        window.title = "Dead or Alive 3"
        window.delegate = self
        if !costumes.isEmpty {
            let items = ["Stock"] + costumes
            func index(_ key: String) -> Int {
                guard let name = settings[key] as? String, let i = costumes.firstIndex(of: name) else { return 0 }
                return i + 1
            }
            addRow("Player 1", costumeP1, items, index("costume_p1"), 236)
            addRow("Player 2", costumeP2, items, index("costume_p2"), 204)
        }
        let savedHeight = intSetting("height", screenHeight)
        let nearest = heights.lastIndex(where: { $0 <= savedHeight }) ?? 0
        addRow("Resolution", resolution,
               heights.map { $0 == 480 ? "480p (windowed)" : "\($0)p" }, nearest, 172)
        let savedMsaa = intSetting("msaa", 1)
        addRow("MSAA", msaa, samples.map { $0 == 1 ? "Off" : "\($0)x" },
               samples.firstIndex(of: savedMsaa) ?? 0, 140)
        smaa.frame = NSRect(x: 104, y: 114, width: 140, height: 20)
        smaa.title = "SMAA (unsupported)"
        smaa.state = .off
        smaa.isEnabled = false
        addRow("Volume", volume, volumes.map { $0 == 0 ? "Mute" : "\($0)%" },
               volumes.firstIndex(of: intSetting("volume", 100)) ?? 0, 80)
        widescreen.frame = NSRect(x: 104, y: 54, width: 150, height: 20)
        widescreen.state = boolSetting("widescreen") ? .on : .off
        widescreen.toolTip = "Wider 3D view. The HUD and menus are stretched."

        let play = NSButton(title: "Play", target: self, action: #selector(play))
        play.frame = NSRect(x: 160, y: 12, width: 88, height: 30)
        play.keyEquivalent = "\r"
        for view in [smaa, widescreen, play] { window.contentView!.addSubview(view) }

        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc func play() {
        func costume(_ popup: NSPopUpButton) -> String {
            popup.indexOfSelectedItem > 0 ? costumes[popup.indexOfSelectedItem - 1] : ""
        }
        let chosen: [String: Any] = [
            "costume_p1": costumes.isEmpty ? "" : costume(costumeP1),
            "costume_p2": costumes.isEmpty ? "" : costume(costumeP2),
            "height": heights[resolution.indexOfSelectedItem],
            "msaa": samples[msaa.indexOfSelectedItem],
            "smaa": false,
            "volume": volumes[volume.indexOfSelectedItem],
            "widescreen": widescreen.state == .on,
        ]
        do {
            try FileManager.default.createDirectory(
                at: settingsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try JSONSerialization.data(
                withJSONObject: chosen, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: settingsURL)
        } catch {
            NSAlert(error: error).runModal()
            return
        }
        launched = true
        NSApp.terminate(nil)
    }

    func windowWillClose(_ notification: Notification) {
        if !launched { exit(1) }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let launcher = Launcher()
app.delegate = launcher
app.run()
exit(launcher.launched ? 0 : 1)
