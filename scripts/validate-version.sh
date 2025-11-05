#!/usr/bin/env bash
# Validate version increment for bump-release

set -e

NEW_VERSION="$1"

# Get current version from pyproject.toml
CURRENT=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

# Parse current version
CURRENT_MAJOR=$(echo "$CURRENT" | cut -d. -f1)
CURRENT_MINOR=$(echo "$CURRENT" | cut -d. -f2)
CURRENT_PATCH=$(echo "$CURRENT" | cut -d. -f3)

# Parse new version
NEW_MAJOR=$(echo "$NEW_VERSION" | cut -d. -f1)
NEW_MINOR=$(echo "$NEW_VERSION" | cut -d. -f2)
NEW_PATCH=$(echo "$NEW_VERSION" | cut -d. -f3)

# Calculate next valid versions
NEXT_MAJOR=$((CURRENT_MAJOR + 1))
NEXT_MINOR=$((CURRENT_MINOR + 1))
NEXT_PATCH=$((CURRENT_PATCH + 1))

# Check if increment is valid
VALID=false

# Major bump: X+1.0.0
if [ "$NEW_MAJOR" -eq "$NEXT_MAJOR" ] && [ "$NEW_MINOR" -eq 0 ] && [ "$NEW_PATCH" -eq 0 ]; then
    VALID=true
# Minor bump: X.Y+1.0
elif [ "$NEW_MAJOR" -eq "$CURRENT_MAJOR" ] && [ "$NEW_MINOR" -eq "$NEXT_MINOR" ] && [ "$NEW_PATCH" -eq 0 ]; then
    VALID=true
# Patch bump: X.Y.Z+1
elif [ "$NEW_MAJOR" -eq "$CURRENT_MAJOR" ] && [ "$NEW_MINOR" -eq "$CURRENT_MINOR" ] && [ "$NEW_PATCH" -eq "$NEXT_PATCH" ]; then
    VALID=true
fi

if [ "$VALID" = "false" ]; then
    echo "❌ Error: Invalid version increment"
    echo "   Current version: $CURRENT"
    echo "   Proposed version: $NEW_VERSION"
    echo ""
    echo "   Valid increments from $CURRENT:"
    echo "   - Major: ${NEXT_MAJOR}.0.0"
    echo "   - Minor: ${CURRENT_MAJOR}.${NEXT_MINOR}.0"
    echo "   - Patch: ${CURRENT_MAJOR}.${CURRENT_MINOR}.${NEXT_PATCH}"
    exit 1
fi

echo "✓ Version increment is valid (${CURRENT} -> ${NEW_VERSION})"
