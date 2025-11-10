#!/bin/bash

# Setup git remote and push
# Usage: ./setup_remote.sh <repository-url>
# Example: ./setup_remote.sh https://github.com/username/repo-name.git

if [ -z "$1" ]; then
    echo "Usage: $0 <repository-url>"
    echo "Example: $0 https://github.com/username/repo-name.git"
    exit 1
fi

REPO_URL=$1

echo "Adding remote repository: $REPO_URL"
git remote add origin "$REPO_URL"

echo "Pushing to remote repository..."
git branch -M main
git push -u origin main

echo "✓ Successfully pushed to remote repository!"

