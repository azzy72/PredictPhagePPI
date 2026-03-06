#!/bin/bash
# Check if dependencies in requirements.txt are available and install them if missing.

# Path to your requirements file relative to this script's location in ./scripts/
REQ_FILE="../requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
    echo "❌ Error: $REQ_FILE not found!"
    exit 1
fi

while read -r line; do
  # Skip empty lines and lines starting with #
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  
  # Extract the package name (handles 'package==version' or just 'package')
  pkg=$(echo "$line" | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1 | xargs)
  
  if pip3 show "$pkg" > /dev/null 2>&1; then
    echo "✅ $pkg is already installed"
  else
    echo "⌛ $pkg is MISSING. Attempting to install..."
    
    # Try to install the package
    if pip3 install "$pkg"; then
      echo "Successfully installed $pkg"
    else
      echo "❌ Failed to install $pkg"
    fi
  fi
done < "$REQ_FILE"