#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import argparse
import os
import re
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk


DEFAULT_AUTOCONNECT_SCRIPT = Path("/usr/local/bin/bt-headset-autoconnect.sh")
MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
DEVICE_LINE_RE = re.compile(
    r"Device\s+((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})(?:\s+\([^)]+\))?\s+(.+)$"
)
DEVICE_NAME_FIELD_RE = re.compile(
    r"Device\s+((?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})\s+(?:Name|Alias):\s+(.+)$"
)


def normalize_mac(mac):
    match = MAC_RE.search(mac or "")
    if not match:
        raise ValueError(f"Invalid Bluetooth MAC: {mac}")
    return match.group(0).upper()


def mac_to_underscore(mac):
    return normalize_mac(mac).replace(":", "_")


def display_device_name(name, mac):
    ascii_name = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in name or "")
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip()
    if ascii_name:
        return ascii_name
    return f"Bluetooth Device {normalize_mac(mac)[-5:]}"


def parse_device_line(line):
    line = ANSI_RE.sub("", line or "").replace("\r", "").strip()
    if not line:
        return None

    field_match = DEVICE_NAME_FIELD_RE.search(line)
    if field_match:
        return normalize_mac(field_match.group(1)), field_match.group(2).strip()

    line_match = DEVICE_LINE_RE.search(line)
    if not line_match:
        return None

    name = line_match.group(2).strip()
    field_name = name.split(":", 1)[0].strip()
    if field_name in {
        "RSSI",
        "TxPower",
        "Class",
        "Icon",
        "Paired",
        "Bonded",
        "Trusted",
        "Blocked",
        "Connected",
        "LegacyPairing",
        "UUIDs",
        "Modalias",
        "ManufacturerData",
        "ServiceData",
    }:
        return None

    return normalize_mac(line_match.group(1)), name


def run_command(command, timeout=20, check=False, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=merged_env,
    )
    if check and result.returncode != 0:
        output = (result.stdout or "").strip()
        raise RuntimeError(f"{' '.join(command)} failed ({result.returncode})\n{output}")
    return result


def bluetoothctl(*args, timeout=20, check=False):
    return run_command(["bluetoothctl", *args], timeout=timeout, check=check)


def update_autoconnect_script(script_path, mac):
    mac = normalize_mac(mac)
    mac_underscore = mac_to_underscore(mac)
    script_path = Path(script_path)
    text = script_path.read_text(encoding="utf-8")

    text, mac_count = re.subn(
        r'^HEADSET_MAC="[^"]*"$',
        f'HEADSET_MAC="{mac}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if mac_count == 0:
        raise RuntimeError(f"HEADSET_MAC not found in {script_path}")

    text, underscore_count = re.subn(
        r'^HEADSET_MAC_UNDERSCORE="[^"]*"$',
        f'HEADSET_MAC_UNDERSCORE="{mac_underscore}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if underscore_count == 0:
        text = re.sub(
            r'^(HEADSET_MAC="[^"]*"\n)',
            rf'\1HEADSET_MAC_UNDERSCORE="{mac_underscore}"\n',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    current_mode = stat.S_IMODE(script_path.stat().st_mode)
    tmp_path = script_path.with_name(f".{script_path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.chmod(tmp_path, current_mode)
    os.replace(tmp_path, script_path)


class BluetoothHeadsetGui(Gtk.Window):
    def __init__(self, args):
        super().__init__(title="Bluetooth Headset")
        self.args = args
        self.devices = {}
        self.devices_lock = threading.Lock()
        self.scan_timer_id = None
        self.scan_process = None
        self.scan_reader_thread = None
        self.refresh_in_progress = False
        self.scanning = False
        self.busy = False
        self.connected = False

        self.set_default_size(1024, 600)
        self.set_border_width(0)
        self.connect("destroy", self.on_destroy)

        self.build_ui()
        self.apply_css()

        if not args.windowed:
            self.fullscreen()

        self.show_all()
        self.run_worker(self.initialize_and_scan)

    def build_ui(self):
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(root)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        header.get_style_context().add_class("header")
        root.pack_start(header, False, False, 0)

        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        header.pack_start(title_box, True, True, 0)

        title = Gtk.Label(label="Bluetooth Headset")
        title.set_xalign(0)
        title.get_style_context().add_class("title")
        title_box.pack_start(title, False, False, 0)

        self.status_label = Gtk.Label(label="Initializing Bluetooth...")
        self.status_label.set_xalign(0)
        self.status_label.get_style_context().add_class("status")
        title_box.pack_start(self.status_label, False, False, 0)

        self.rescan_button = Gtk.Button(label="Scan")
        self.rescan_button.connect("clicked", self.on_rescan_clicked)
        header.pack_start(self.rescan_button, False, False, 0)

        self.exit_button = Gtk.Button(label="Exit")
        self.exit_button.connect("clicked", lambda _button: Gtk.main_quit())
        header.pack_start(self.exit_button, False, False, 0)

        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        root.pack_start(content, True, True, 0)

        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        left.get_style_context().add_class("device-pane")
        content.pack_start(left, True, True, 0)

        hint = Gtk.Label(label="Put your headset in pairing mode, then click its name in the list.")
        hint.set_xalign(0)
        hint.get_style_context().add_class("hint")
        left.pack_start(hint, False, False, 0)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        left.pack_start(scroller, True, True, 0)

        self.list_box = Gtk.ListBox()
        self.list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.list_box.set_activate_on_single_click(True)
        self.list_box.connect("row-activated", self.on_row_activated)
        scroller.add(self.list_box)

        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        right.get_style_context().add_class("log-pane")
        content.pack_start(right, False, False, 0)

        log_title = Gtk.Label(label="Status")
        log_title.set_xalign(0)
        log_title.get_style_context().add_class("log-title")
        right.pack_start(log_title, False, False, 0)

        log_scroller = Gtk.ScrolledWindow()
        log_scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        right.pack_start(log_scroller, True, True, 0)

        self.log_view = Gtk.TextView()
        self.log_view.set_editable(False)
        self.log_view.set_cursor_visible(False)
        self.log_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.log_buffer = self.log_view.get_buffer()
        log_scroller.add(self.log_view)

    def apply_css(self):
        css = b"""
        window {
            background: #f4f7fb;
            color: #1f2933;
            font-family: sans-serif;
        }
        .header {
            background: #1d4f73;
            padding: 22px 28px;
            color: white;
        }
        .title {
            color: white;
            font-size: 34px;
            font-weight: 700;
        }
        .status {
            color: #d9edf7;
            font-size: 18px;
        }
        button {
            min-width: 112px;
            min-height: 52px;
            font-size: 20px;
            border-radius: 6px;
            padding: 8px 16px;
        }
        .device-pane {
            padding: 18px 22px;
        }
        .hint {
            color: #52606d;
            font-size: 18px;
            padding: 0 0 14px 2px;
        }
        .device-row {
            background: white;
            border: 1px solid #d7dde4;
            border-radius: 6px;
            margin: 0 0 10px 0;
            padding: 16px 18px;
        }
        .device-name {
            color: #102a43;
            font-size: 26px;
            font-weight: 700;
        }
        .device-mac {
            color: #627d98;
            font-size: 18px;
        }
        .log-pane {
            background: #e8eef5;
            border-left: 1px solid #c8d2de;
            min-width: 360px;
            padding: 18px;
        }
        .log-title {
            color: #243b53;
            font-size: 22px;
            font-weight: 700;
        }
        textview {
            font-size: 14px;
            background: #ffffff;
            color: #1f2933;
        }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            self.get_screen(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def set_status(self, message):
        self.status_label.set_text(message)

    def append_log(self, message):
        stamp = time.strftime("%H:%M:%S")
        end_iter = self.log_buffer.get_end_iter()
        self.log_buffer.insert(end_iter, f"[{stamp}] {message}\n")
        mark = self.log_buffer.create_mark(None, self.log_buffer.get_end_iter(), False)
        self.log_view.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)

    def ui(self, func, *args):
        GLib.idle_add(func, *args)

    def run_worker(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()

    def set_busy(self, busy):
        self.busy = busy
        self.rescan_button.set_sensitive(not busy)
        self.list_box.set_sensitive(not busy)

    def initialize_and_scan(self):
        self.ui(self.set_busy, True)
        try:
            self.log("Initializing Bluetooth controller")
            self.init_bluetooth_environment()
            self.log("Bluetooth initialization complete")
            self.ui(self.set_status, "Scanning for Bluetooth devices...")
            self.start_scan()
        except Exception as exc:
            self.ui(self.set_status, "Bluetooth initialization failed")
            self.log(f"Error: {exc}")
        finally:
            self.ui(self.set_busy, False)

    def init_bluetooth_environment(self):
        self.safe_run(["modprobe", "moal", "mod_para=nxp/wifi_mod_para.conf"], timeout=10)
        time.sleep(1)
        if self.safe_run(["ifconfig", "mlan0", "up"], timeout=8).returncode != 0:
            self.safe_run(["ip", "link", "set", "mlan0", "up"], timeout=8)

        self.safe_run(["modprobe", "btnxpuart"], timeout=10)
        self.wait_for_hci0()

        self.safe_run(["hciconfig", "hci0", "reset"], timeout=10)
        time.sleep(1)
        self.safe_run(["hciconfig", "hci0", "up"], timeout=10)
        self.safe_run(["systemctl", "start", "bluetooth"], timeout=20)
        self.safe_run(["systemctl", "enable", "bluetooth"], timeout=20)
        time.sleep(2)

        bluetoothctl("power", "on", timeout=10)
        bluetoothctl("pairable", "on", timeout=10)
        bluetoothctl("agent", "on", timeout=10)
        bluetoothctl("default-agent", timeout=10)

    def wait_for_hci0(self):
        for _index in range(20):
            if Path("/sys/class/bluetooth/hci0").exists():
                return
            time.sleep(1)
        raise RuntimeError("Timed out waiting for hci0. Check the btnxpuart driver.")

    def safe_run(self, command, timeout=20):
        try:
            result = run_command(command, timeout=timeout)
            if result.returncode != 0:
                output = (result.stdout or "").strip()
                self.log(f"Ignored failure: {' '.join(command)} {output}")
            return result
        except FileNotFoundError:
            self.log(f"Command not found: {command[0]}")
            return subprocess.CompletedProcess(command, 127, "")
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out: {' '.join(command)}")
            return subprocess.CompletedProcess(command, 124, "")

    def start_scan(self):
        self.stop_scan()
        self.scanning = True
        try:
            self.start_scan_process()
        except Exception as exc:
            self.log(f"Failed to start scan: {exc}")
        self.ui(self.set_status, "Scanning. Click a headset name to connect.")
        self.ui(self.refresh_devices)
        self.ui(self.install_scan_timer)

    def start_scan_process(self):
        self.scan_process = subprocess.Popen(
            ["bluetoothctl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.scan_reader_thread = threading.Thread(
            target=self.read_scan_output,
            daemon=True,
        )
        self.scan_reader_thread.start()
        self.scan_process.stdin.write("scan on\n")
        self.scan_process.stdin.flush()

    def read_scan_output(self):
        process = self.scan_process
        if not process or not process.stdout:
            return
        for line in process.stdout:
            parsed = parse_device_line(line)
            if not parsed:
                continue
            mac, name = parsed
            self.add_or_update_device(mac, name, source="scan")

    def install_scan_timer(self):
        if self.scan_timer_id:
            GLib.source_remove(self.scan_timer_id)
        self.scan_timer_id = GLib.timeout_add_seconds(
            self.args.scan_interval, self.refresh_devices
        )

    def stop_scan(self):
        self.scanning = False
        if self.scan_timer_id:
            GLib.source_remove(self.scan_timer_id)
            self.scan_timer_id = None
        if self.scan_process:
            process = self.scan_process
            self.scan_process = None
            try:
                if process.stdin and process.poll() is None:
                    process.stdin.write("scan off\nquit\n")
                    process.stdin.flush()
            except Exception:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
        try:
            bluetoothctl("scan", "off", timeout=10)
        except Exception as exc:
            self.log(f"Failed to stop scan: {exc}")

    def refresh_devices(self):
        if self.busy or self.refresh_in_progress:
            return True
        self.refresh_in_progress = True
        self.run_worker(self.refresh_devices_worker)
        return True

    def refresh_devices_worker(self):
        try:
            devices = self.read_devices()
            for mac, name in devices.items():
                self.add_or_update_device(mac, name, source="cache", log_new=False)
            with self.devices_lock:
                device_count = len(self.devices)
            if device_count:
                self.ui(self.set_status, f"Found {device_count} device(s). Click a headset to connect.")
            else:
                self.ui(self.set_status, "Scanning. No devices found yet.")
        except Exception as exc:
            self.log(f"Failed to refresh devices: {exc}")
        finally:
            self.refresh_in_progress = False

    def read_devices(self):
        result = bluetoothctl("devices", timeout=10)
        devices = {}
        for line in (result.stdout or "").splitlines():
            parsed = parse_device_line(line)
            if not parsed:
                continue
            mac, name = parsed
            name = name or "Unknown Device"
            devices[mac] = name
        return devices

    def add_or_update_device(self, mac, name, source, log_new=True):
        mac = normalize_mac(mac)
        name = name or "Unknown Device"
        changed = False
        is_new = False
        with self.devices_lock:
            old_name = self.devices.get(mac)
            if old_name is None:
                self.devices[mac] = name
                changed = True
                is_new = True
            elif old_name != name and display_device_name(name, mac) != display_device_name(old_name, mac):
                self.devices[mac] = name
                changed = True

            device_count = len(self.devices)

        if changed:
            if is_new and log_new:
                self.log(f"Found device: {display_device_name(name, mac)} {mac}")
            self.ui(self.render_devices)
            self.ui(self.set_status, f"Found {device_count} device(s). Click a headset to connect.")

    def render_devices(self):
        for child in self.list_box.get_children():
            self.list_box.remove(child)

        with self.devices_lock:
            devices = sorted(self.devices.items(), key=lambda item: display_device_name(item[1], item[0]).lower())

        for mac, name in devices:
            row = Gtk.ListBoxRow()
            row.mac = mac
            row.name = display_device_name(name, mac)
            row.get_style_context().add_class("device-row")

            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            row.add(box)

            name_label = Gtk.Label(label=row.name)
            name_label.set_xalign(0)
            name_label.set_ellipsize(3)
            name_label.get_style_context().add_class("device-name")
            box.pack_start(name_label, False, False, 0)

            mac_label = Gtk.Label(label=mac)
            mac_label.set_xalign(0)
            mac_label.get_style_context().add_class("device-mac")
            box.pack_start(mac_label, False, False, 0)

            self.list_box.add(row)

        self.list_box.show_all()

    def on_row_activated(self, _list_box, row):
        if self.busy:
            return
        self.run_worker(self.connect_device_worker, row.mac, row.name)

    def on_rescan_clicked(self, _button):
        if self.busy:
            return
        with self.devices_lock:
            self.devices.clear()
        self.render_devices()
        self.run_worker(self.initialize_and_scan)

    def connect_device_worker(self, mac, name):
        self.ui(self.set_busy, True)
        self.ui(self.set_status, f"Connecting to {display_device_name(name, mac)}...")
        try:
            self.stop_scan()
            self.log(f"Selected device: {display_device_name(name, mac)} {mac}")
            self.disconnect_existing_devices(except_mac=mac)
            self.pair_and_trust(mac)
            self.log("Updating headset MAC in autoconnect script")
            update_autoconnect_script(self.args.autoconnect_script, mac)
            self.log("Running headset autoconnect script")
            result = run_command(
                [str(self.args.autoconnect_script)],
                timeout=self.args.connect_timeout,
            )
            output = (result.stdout or "").strip()
            if output:
                self.log(output[-3000:])
            if result.returncode != 0:
                raise RuntimeError(f"Autoconnect script failed with exit code {result.returncode}")
            self.connected = True
            self.ui(self.set_status, "Connected. Closing Bluetooth setup.")
            self.log("Connected")
            self.ui(self.schedule_success_quit)
        except Exception as exc:
            self.ui(self.set_status, "Connection failed. Select a device and try again.")
            self.log(f"Error: {exc}")
            self.start_scan()
        finally:
            self.ui(self.set_busy, False)

    def disconnect_existing_devices(self, except_mac=None):
        except_mac = normalize_mac(except_mac) if except_mac else None
        devices = self.read_devices()
        for mac in devices:
            if mac == except_mac:
                continue
            info = bluetoothctl("info", mac, timeout=8).stdout or ""
            if "Connected: yes" in info:
                self.log(f"Disconnecting old device: {mac}")
                bluetoothctl("disconnect", mac, timeout=15)

    def pair_and_trust(self, mac):
        mac = normalize_mac(mac)
        self.log(f"Pairing: {mac}")
        pair_result = bluetoothctl("pair", mac, timeout=45)
        pair_output = pair_result.stdout or ""
        info = bluetoothctl("info", mac, timeout=10).stdout or ""
        if pair_result.returncode != 0 and "Paired: yes" not in info:
            if "AlreadyExists" not in pair_output and "already exists" not in pair_output.lower():
                raise RuntimeError(f"Pair failed: {pair_output.strip()}")

        self.log(f"Trusting device: {mac}")
        trust_result = bluetoothctl("trust", mac, timeout=15)
        if trust_result.returncode != 0:
            raise RuntimeError(f"Trust failed: {(trust_result.stdout or '').strip()}")

    def log(self, message):
        self.ui(self.append_log, message)

    def schedule_success_quit(self):
        GLib.timeout_add_seconds(2, Gtk.main_quit)
        return False

    def on_destroy(self, _window):
        try:
            self.stop_scan()
        finally:
            Gtk.main_quit()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Bluetooth headset selection GUI")
    parser.add_argument(
        "--autoconnect-script",
        type=Path,
        default=DEFAULT_AUTOCONNECT_SCRIPT,
        help="Path to bt-headset-autoconnect.sh",
    )
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=3,
        help="Device list refresh interval in seconds",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=180,
        help="Timeout for the autoconnect script",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Do not fullscreen the GTK window",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    if not args.autoconnect_script.exists():
        print(f"Autoconnect script not found: {args.autoconnect_script}", file=sys.stderr)
        return 2
    window = BluetoothHeadsetGui(args)
    Gtk.main()
    return 0 if window.connected else 1


if __name__ == "__main__":
    raise SystemExit(main())
