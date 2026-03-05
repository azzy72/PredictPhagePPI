#!/bin/bash
# Check if dependencies in requirements.txt are available to the current environment.

while read line; do
  pkg=$(echo $line | cut -d'=' -f1)
  if pip show $pkg > /dev/null 2>&1; then
    echo "✅ $pkg is installed"
  else
    echo "❌ $pkg is MISSING"
  fi
done < ../requirements.txt