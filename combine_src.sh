#!/bin/bash

# Project root directory and source directory
PROJECT_DIR="$(dirname "$0")"
SRC_DIR="$PROJECT_DIR/src"
PACKAGE_FILE="$PROJECT_DIR/src/GeneralizedRFFs.jl"
COMBINE_FILE="$PROJECT_DIR/combined_src.jl"

echo "# Generated combined_src.jl" > "$COMBINE_FILE"
echo "" >> "$COMBINE_FILE"

while IFS= read -r line; do
    # detect rows with include("*.jl")
    if [[ $line =~ include\(\"([^\"]+)\"\) ]]; then
        FILE_NAME="${BASH_REMATCH[1]}"
        FILE_PATH="$SRC_DIR/$FILE_NAME"
        
        if [[ -f "$FILE_PATH" ]]; then
            echo "# ---- $FILE_NAME ----" >> "$COMBINE_FILE"
            cat "$FILE_PATH" >> "$COMBINE_FILE"
            echo "" >> "$COMBINE_FILE"
        else
            echo "Warning: File not found - $FILE_NAME" >&2
        fi
    else
        echo "$line" >> "$COMBINE_FILE"
    fi
done < "$PACKAGE_FILE"

echo "combined_src.jl has been generated."