#!/bin/bash
#
# Initialize a new project from an exemplar template.
#
# Usage: ./init-from-exemplar.sh <exemplar-name> <new-project-name> [destination-dir]
#
# Examples:
#   ./init-from-exemplar.sh batch-etl-pipeline my-etl-project
#   ./init-from-exemplar.sh streaming-lakehouse my-streaming-app ~/projects
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory (to find exemplars relative to it)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXEMPLARS_DIR="$REPO_ROOT/exemplars"

# Parse arguments
EXEMPLAR_NAME="$1"
PROJECT_NAME="$2"
DEST_DIR="${3:-$(pwd)}"

# Validate arguments
if [ -z "$EXEMPLAR_NAME" ] || [ -z "$PROJECT_NAME" ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo ""
    echo "Usage: $0 <exemplar-name> <new-project-name> [destination-dir]"
    echo ""
    echo "Available exemplars:"
    ls -1 "$EXEMPLARS_DIR" | grep -v README.md | sed 's/^/  - /'
    exit 1
fi

# Check exemplar exists
EXEMPLAR_PATH="$EXEMPLARS_DIR/$EXEMPLAR_NAME"
if [ ! -d "$EXEMPLAR_PATH" ]; then
    echo -e "${RED}Error: Exemplar '$EXEMPLAR_NAME' not found${NC}"
    echo ""
    echo "Available exemplars:"
    ls -1 "$EXEMPLARS_DIR" | grep -v README.md | sed 's/^/  - /'
    exit 1
fi

# Set up destination
PROJECT_PATH="$DEST_DIR/$PROJECT_NAME"

# Check destination doesn't already exist
if [ -d "$PROJECT_PATH" ]; then
    echo -e "${RED}Error: Destination '$PROJECT_PATH' already exists${NC}"
    exit 1
fi

echo -e "${GREEN}Initializing new project from exemplar...${NC}"
echo ""
echo "  Exemplar:    $EXEMPLAR_NAME"
echo "  Project:     $PROJECT_NAME"
echo "  Destination: $PROJECT_PATH"
echo ""

# Copy exemplar
echo -e "${YELLOW}Copying exemplar files...${NC}"
cp -r "$EXEMPLAR_PATH" "$PROJECT_PATH"

# Update bundle name in databricks.yml
echo -e "${YELLOW}Updating bundle name...${NC}"
if [ -f "$PROJECT_PATH/databricks.yml" ]; then
    # Use sed to replace bundle name (works on both macOS and Linux)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/name: $EXEMPLAR_NAME/name: $PROJECT_NAME/g" "$PROJECT_PATH/databricks.yml"
    else
        sed -i "s/name: $EXEMPLAR_NAME/name: $PROJECT_NAME/g" "$PROJECT_PATH/databricks.yml"
    fi
fi

# Initialize git repository (optional)
echo -e "${YELLOW}Initializing git repository...${NC}"
cd "$PROJECT_PATH"
git init --quiet

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# Databricks
.databricks/

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db
EOF
fi

echo ""
echo -e "${GREEN}✓ Project initialized successfully!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. cd $PROJECT_PATH"
echo ""
echo "  2. Update databricks.yml with your settings:"
echo "     - Set your catalog and schema names"
echo "     - Configure workspace host (or set DATABRICKS_HOST env var)"
echo ""
echo "  3. Customize the code in src/ for your use case"
echo ""
echo "  4. Deploy to Databricks:"
echo "     databricks bundle validate"
echo "     databricks bundle deploy"
echo ""
echo "See SETUP.md for detailed instructions."
