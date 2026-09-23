#!/usr/bin/env bash
# Drive a RunPod pod over the interactive-only SSH proxy.
#
# The proxy at ssh.runpod.io accepts no exec-form commands and no SCP: an
# `ssh host 'cmd'` invocation authenticates and then hangs forever. The only
# thing that works is an interactive shell with a forced PTY and commands fed
# on stdin, so this wrapper pipes a heredoc in and strips the terminal control
# bytes back out.
#
# Usage:
#   scripts/rp.sh <<'EOF'          # commands on stdin
#   nvidia-smi
#   EOF
#   echo "nvidia-smi" | scripts/rp.sh
#   scripts/rp.sh -c "nvidia-smi"  # single command
#
# Env:
#   RP_HOST   proxy login, e.g. t79vcyoczutmkp-64411c54@ssh.runpod.io
#   RP_KEY    identity file (default ~/.ssh/id_ed25519)
#   RP_TMO    seconds to wait for the sentinel before giving up (default 900)
set -uo pipefail

RP_HOST="${RP_HOST:-}"
RP_KEY="${RP_KEY:-$HOME/.ssh/id_ed25519}"
RP_TMO="${RP_TMO:-900}"

if [ -z "$RP_HOST" ]; then
  echo "rp.sh: set RP_HOST to <podid>-<user>@ssh.runpod.io" >&2
  exit 2
fi

if [ "${1:-}" = "-c" ]; then
  shift
  payload="$*"
else
  payload="$(cat)"
fi

# A sentinel lets us detect completion and cut the interactive banner/prompt
# noise, since the proxy gives us a login shell rather than a clean pipe.
# Split into two halves so the echoed source line never contains the assembled
# sentinel; only the shell's output does.
BEGIN_L="RPBEG"; BEGIN_R="IN-$$"
END_L="RPE"; END_R="ND-$$"
BEGIN="${BEGIN_L}${BEGIN_R}"
END="${END_L}${END_R}"

# The PTY echoes every line we send, so a sentinel written literally would
# appear twice: once as the echoed source and once as real output. Splitting the
# literal in the source means only the output side matches.
script="$(
  printf 'export PS1=\n'
  printf 'stty -echo 2>/dev/null || true\n'
  printf 'echo "%s""%s"\n' "$BEGIN_L" "$BEGIN_R"
  printf '%s\n' "$payload"
  printf 'echo "%s""%s" rc=$?\n' "$END_L" "$END_R"
  printf 'exit\n'
)"

raw="$(
  printf '%s\n' "$script" | ssh -tt \
    -o ConnectTimeout=25 \
    -o StrictHostKeyChecking=accept-new \
    -o IdentitiesOnly=yes \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=6 \
    -i "$RP_KEY" \
    "$RP_HOST" 2>&1
)"
ssh_rc=$?

# Strip CR and ANSI escapes, then keep only what is between the sentinels.
clean="$(printf '%s' "$raw" | tr -d '\r' | sed -E 's/\x1B\[[0-9;?]*[a-zA-Z]//g; s/\x1B\][^\x07]*\x07//g')"

if printf '%s' "$clean" | grep -q "$BEGIN"; then
  body="$(printf '%s' "$clean" | sed -n "/$BEGIN/,/$END/p" | sed "1d;/$END/d")"
  printf '%s\n' "$body"
  rc="$(printf '%s' "$clean" | grep -o "$END rc=[0-9]*" | head -1 | grep -o '[0-9]*$')"
  if [ -z "$rc" ]; then
    # Commands started but the closing sentinel never arrived, so the session
    # was cut short. Reporting success here would hide a truncated run.
    echo "rp.sh: session truncated before completion (no end sentinel)" >&2
    exit 1
  fi
  exit "$rc"
fi

# No sentinel: the session never reached our commands. Show the raw transcript
# so the failure mode (auth, PTY, dead sshd) is visible rather than silent.
echo "rp.sh: session produced no sentinel (ssh rc=$ssh_rc); raw transcript:" >&2
printf '%s\n' "$clean" >&2
exit 1
