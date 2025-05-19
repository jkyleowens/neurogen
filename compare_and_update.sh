#!/bin/bash

# Function to compare files and update if needed
compare_and_update() {
    local source_file="$1"
    local dest_file="$2"
    
    if [ ! -f "$source_file" ]; then
        echo "Source file $source_file does not exist"
        return 1
    fi
    
    if [ ! -f "$dest_file" ]; then
        echo "Destination file $dest_file does not exist, copying source file"
        cp "$source_file" "$dest_file"
        return 0
    fi
    
    # Compare files
    if diff -q "$source_file" "$dest_file" > /dev/null; then
        echo "Files $source_file and $dest_file are identical"
    else
        echo "Files $source_file and $dest_file are different"
        echo "Creating backup of $dest_file"
        cp "$dest_file" "${dest_file}.bak"
        echo "Updating $dest_file with content from $source_file"
        cp "$source_file" "$dest_file"
    fi
}

# Compare and update reset_model_state.py
compare_and_update "Uploads/reset_model_state.py" "src/utils/reset_model_state.py"

# Compare and update tensor_shape_fixes.py
compare_and_update "Uploads/tensor_shape_fixes.py" "src/utils/tensor_shape_fixes.py"

# Copy README.md and requirements.txt from Uploads to project root if they're different
compare_and_update "Uploads/README.md" "README.md"
compare_and_update "Uploads/requirements.txt" "requirements.txt"
