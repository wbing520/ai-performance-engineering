#!/bin/bash

# Architecture Switching Script
# Switch between Hopper H100/H200 and Blackwell B200/B300

set -e

ARCH="$1"

if [ -z "$ARCH" ]; then
    echo "Usage: $0 [sm_90|sm_100]"
    echo "  sm_90  - Hopper H100/H200"
    echo "  sm_100 - Blackwell B200/B300"
    exit 1
fi

if [ "$ARCH" != "sm_90" ] && [ "$ARCH" != "sm_100" ]; then
    echo "Invalid architecture: $ARCH"
    echo "Valid options: sm_90, sm_100"
    exit 1
fi

echo "Switching to architecture: $ARCH"

# Update all Makefiles
find . -name "Makefile" -type f | while read -r makefile; do
    echo "Updating $makefile..."
    sed -i.bak "s/ARCH ?= sm_[0-9]*/ARCH ?= $ARCH/g" "$makefile"
    rm -f "$makefile.bak"
done

# Rebuild all projects
echo "Rebuilding all projects..."
find code -name "Makefile" -type f | while read -r makefile; do
    dir=$(dirname "$makefile")
    echo "Rebuilding $dir..."
    cd "$dir"
    make clean
    make ARCH=$ARCH
    cd - > /dev/null
done

echo "Architecture switched to $ARCH"
