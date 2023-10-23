#!/bin/bash

# Use the $HOME environment variable to get the user's home directory
home_dir=$HOME

# Construct the path to the Downloads folder
download_dir="$home_dir/Downloads"

public_dir="public"

# Find the last file containing "bunka_docs"
last_bunka_docs_file=$(find "$download_dir" -type f -name "*bunka_docs*" | sort -r | head -n 1)

# Find the last file containing "bunka_topics"
last_bunka_topics_file=$(find "$download_dir" -type f -name "*bunka_topics*" | sort -r | head -n 1)

# Check if matching files were found for both bunka_docs and bunka_topics
if [ -n "$last_bunka_docs_file" ] && [ -n "$last_bunka_topics_file" ]; then
  # Use mv to move the files to the /public directory
  mv "$last_bunka_docs_file" "$public_dir/"
  mv "$last_bunka_topics_file" "$public_dir/"
  echo "Moved $last_bunka_docs_file and $last_bunka_topics_file to $public_dir/"
else
  echo "No matching files found in $download_dir for bunka_docs and/or bunka_topics."
fi
