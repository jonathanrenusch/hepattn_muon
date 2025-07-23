#!/bin/bash
# Setup script to configure pixi cache directory
# This avoids AFS quota issues by using local storage

export PIXI_CACHE_DIR=/shared/pixi_cache
export PIXI_HOME=/shared/pixi_cache

echo "Pixi cache configured:"
echo "  PIXI_CACHE_DIR: $PIXI_CACHE_DIR"
echo "  PIXI_HOME: $PIXI_HOME"
echo ""
echo "You can now run pixi commands normally."
echo "To make this permanent, add these exports to your ~/.bashrc:"
echo "  export PIXI_CACHE_DIR=/shared/pixi_cache"
echo "  export PIXI_HOME=/shared/pixi_cache"
