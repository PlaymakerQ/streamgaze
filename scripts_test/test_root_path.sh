#!/bin/bash

echo "1. current script path:"
echo "$0"

echo ""
echo "2. dirname of script:"
echo "$(dirname "$0")"

echo ""
echo "3. parent directory:"
echo "$(dirname "$0")/.."

echo ""
echo "4. absolute root path:"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "$ROOT_DIR"