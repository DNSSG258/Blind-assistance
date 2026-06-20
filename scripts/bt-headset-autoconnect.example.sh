#!/bin/sh
# Example Bluetooth headset setup script for FRDM-IMX93.
# Copy this file to /usr/local/bin/bt-headset-autoconnect.sh, replace
# HEADSET_MAC with the target headset MAC address, then make it executable.

HEADSET_MAC="AA:BB:CC:DD:EE:FF"
HEADSET_MAC_UNDERSCORE="$(echo "$HEADSET_MAC" | tr ':' '_')"
BLUEZ_DEV_PATH="/org/bluez/hci0/dev_${HEADSET_MAC_UNDERSCORE}"
PA_CARD="bluez_card.${HEADSET_MAC_UNDERSCORE}"
LOG_TAG="[bt-headset-auto]"

export XDG_RUNTIME_DIR=/run/user/0
export PULSE_RUNTIME_PATH=/run/user/0/pulse
unset PULSE_SERVER

log()
{
    echo "${LOG_TAG} $*"
}

wait_for_hci0()
{
    log "Waiting for hci0..."
    i=0
    while [ $i -lt 20 ]; do
        if [ -d /sys/class/bluetooth/hci0 ]; then
            log "hci0 is ready"
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    log "ERROR: hci0 timeout"
    return 1
}

stop_conflicting_audio_services()
{
    log "Stopping BlueALSA, PipeWire, and extra PulseAudio processes"
    systemctl stop bluealsa 2>/dev/null || true
    systemctl stop bluealsa-aplay 2>/dev/null || true
    systemctl stop pipewire 2>/dev/null || true
    systemctl stop wireplumber 2>/dev/null || true
    systemctl stop pipewire-pulse 2>/dev/null || true
    systemctl stop pulseaudio 2>/dev/null || true
    pkill -x bluealsa-aplay 2>/dev/null || true
    pkill -x bluealsa 2>/dev/null || true
    pkill -x pipewire 2>/dev/null || true
    pkill -x wireplumber 2>/dev/null || true
    pkill -x pipewire-pulse 2>/dev/null || true
    pkill -u pulse pulseaudio 2>/dev/null || true
    sleep 1
}

restart_root_pulseaudio()
{
    log "Restarting root PulseAudio"
    mkdir -p /run/user/0
    chmod 700 /run/user/0
    pulseaudio -k 2>/dev/null || true
    pkill -u root pulseaudio 2>/dev/null || true
    sleep 2
    pulseaudio --start
    sleep 2
    pactl info >/dev/null 2>&1 || {
        log "ERROR: pactl cannot connect to PulseAudio"
        return 1
    }
}

load_pulseaudio_bluetooth_modules()
{
    log "Loading PulseAudio Bluetooth modules"
    pactl load-module module-bluetooth-policy 2>/dev/null || true
    pactl load-module module-bluetooth-discover 2>/dev/null || true
    sleep 1
}

init_wifi_bt_driver()
{
    log "Loading WiFi/BT drivers"
    modprobe moal mod_para=nxp/wifi_mod_para.conf 2>/dev/null || true
    sleep 1
    ifconfig mlan0 up 2>/dev/null || ip link set mlan0 up 2>/dev/null || true
    modprobe btnxpuart 2>/dev/null || true
    sleep 2
    wait_for_hci0 || return 1
    hciconfig hci0 reset 2>/dev/null || true
    sleep 1
    hciconfig hci0 up 2>/dev/null || true
}

restart_bluetooth_service()
{
    log "Restarting bluetooth service"
    systemctl restart bluetooth 2>/dev/null || /etc/init.d/bluetooth restart 2>/dev/null || true
    sleep 3
    bluetoothctl power on
    bluetoothctl pairable on
    bluetoothctl agent on || true
    bluetoothctl default-agent || true
}

connect_headset()
{
    log "Connecting headset ${HEADSET_MAC}"
    bluetoothctl trust "${HEADSET_MAC}" 2>/dev/null || true
    bluetoothctl disconnect "${HEADSET_MAC}" 2>/dev/null || true
    sleep 2
    i=0
    while [ $i -lt 5 ]; do
        bluetoothctl connect "${HEADSET_MAC}" 2>/dev/null || true
        sleep 3
        if bluetoothctl info "${HEADSET_MAC}" 2>/dev/null | grep -q "Connected: yes"; then
            log "Headset connected"
            return 0
        fi
        i=$((i + 1))
    done
    log "ERROR: headset connection failed"
    return 1
}

wait_for_bluez_card()
{
    log "Waiting for PulseAudio card ${PA_CARD}"
    i=0
    while [ $i -lt 15 ]; do
        pactl list cards short | grep -q "${PA_CARD}" && return 0
        sleep 1
        i=$((i + 1))
    done
    pactl load-module module-bluez5-device path="${BLUEZ_DEV_PATH}" 2>/dev/null || true
    sleep 2
    pactl list cards short | grep -q "${PA_CARD}"
}

set_a2dp_profile()
{
    log "Setting A2DP profile"
    i=0
    while [ $i -lt 5 ]; do
        pactl set-card-profile "${PA_CARD}" a2dp_sink 2>/dev/null && return 0
        bluetoothctl disconnect "${HEADSET_MAC}" 2>/dev/null || true
        sleep 2
        bluetoothctl connect "${HEADSET_MAC}" 2>/dev/null || true
        sleep 4
        i=$((i + 1))
    done
    return 1
}

set_default_sink()
{
    SINK_NAME=""
    i=0
    while [ $i -lt 10 ]; do
        SINK_NAME="$(pactl list short sinks | awk '/bluez_output.*a2dp/ {print $2; exit}')"
        [ -n "${SINK_NAME}" ] && break
        sleep 1
        i=$((i + 1))
    done
    [ -n "${SINK_NAME}" ] || return 1
    log "Setting default sink: ${SINK_NAME}"
    pactl set-default-sink "${SINK_NAME}"
}

main()
{
    log "Starting Bluetooth headset setup"
    init_wifi_bt_driver || exit 1
    restart_bluetooth_service
    stop_conflicting_audio_services
    restart_root_pulseaudio || exit 1
    load_pulseaudio_bluetooth_modules
    connect_headset || exit 1
    wait_for_bluez_card || exit 1
    set_a2dp_profile || exit 1
    set_default_sink || exit 1
    log "Bluetooth headset setup complete"
}

main "$@"
