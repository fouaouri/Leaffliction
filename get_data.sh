#!/bin/bash

echo "Downloading leaves.zip..."
curl -L -o leaves.zip "https://cdn.intra.42.fr/document/document/42036/leaves.zip"
echo "Downloaded successfully!"

mkdir -p leaves
unzip -q leaves.zip -d leaves
echo "Extracted to leaves/"

# Detect the top-level folder created by unzip
topdir=$(find leaves -mindepth 1 -maxdepth 1 -type d | head -n 1)
echo "Top-level extracted directory: $topdir"

# Create Apple and Grape directories
mkdir -p leaves/Apple
mkdir -p leaves/Grape
echo "Created Apple and Grape directories."

# Move directories starting with Apple or Grape from topdir
for dir in "$topdir"/*; do
    if [ -d "$dir" ]; then
        base=$(basename "$dir")
        if [[ "$base" == Apple* ]]; then
            mv "$dir" leaves/Apple/
        elif [[ "$base" == Grape* ]]; then
            mv "$dir" leaves/Grape/
        fi
    fi
done

echo "Moved directories successfully!"

# Optional: remove the remaining extracted folder and zip
rm -rf "$topdir"
rm leaves.zip