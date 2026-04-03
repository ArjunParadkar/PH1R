#!/usr/bin/env bash
# Build or locate Gkeyll for VM Weibel regression; set GKEYLL_BIN.
# Use VM Weibel regression (e.g. Regression/vm-weibel/rt-weibel-2x2v-p2.lua). CPU by default.

set -e
GKEYLL_ROOT="${GKEYLL_ROOT:-$HOME/gkeyll}"
if [ -d "$GKEYLL_ROOT" ]; then
  echo "GKEYLL_ROOT=$GKEYLL_ROOT"
  if [ -x "$GKEYLL_ROOT/build/gkeyll/gkeyll" ]; then
    export GKEYLL_BIN="$GKEYLL_ROOT/build/gkeyll/gkeyll"
    echo "GKEYLL_BIN=$GKEYLL_BIN"
    exit 0
  fi
fi
echo "Gkeyll not found at $GKEYLL_ROOT. Clone and build:"
echo "  git clone https://github.com/ammarhakim/gkyl.git $GKEYLL_ROOT && cd $GKEYLL_ROOT && ./waf configure && ./waf build"
echo "Or set GKEYLL_BIN to existing binary. VM dataset will use synthetic mode amplitudes if Gkeyll unavailable."
