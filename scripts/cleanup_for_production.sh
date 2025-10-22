#!/bin/bash
# Cleanup script for production/GitHub deployment
# Removes unnecessary files while keeping essential ones

echo "Mars Mission AI - Production Cleanup Script"
echo "==========================================="
echo ""

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Working directory: $PROJECT_ROOT"
echo ""

# Confirmation
read -p "This will remove development/test files. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting cleanup..."
echo ""

# Remove Python cache
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name "*.pyd" -delete 2>/dev/null

# Remove pytest cache
echo "Removing pytest cache..."
rm -rf .pytest_cache 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# Remove coverage files
echo "Removing coverage files..."
rm -f .coverage 2>/dev/null
rm -rf htmlcov/ 2>/dev/null
rm -f coverage.xml 2>/dev/null

# Remove virtual environments (keep requirements.txt)
echo "Removing virtual environments..."
rm -rf venv/ 2>/dev/null
rm -rf .venv/ 2>/dev/null
rm -rf env/ 2>/dev/null

# Remove IDE/editor files
echo "Removing IDE files..."
rm -rf .vscode/ 2>/dev/null
rm -rf .idea/ 2>/dev/null
rm -f .DS_Store 2>/dev/null
find . -name ".DS_Store" -delete 2>/dev/null
rm -f *.swp 2>/dev/null
rm -f *.swo 2>/dev/null
rm -f *~ 2>/dev/null

# Remove log files (keep logs directory structure)
echo "Removing log files..."
find logs/ -type f -name "*.log" -delete 2>/dev/null
find logs/ -type f -name "*.jsonl" -delete 2>/dev/null

# Remove temporary files
echo "Removing temporary files..."
rm -rf tmp/ 2>/dev/null
rm -rf temp/ 2>/dev/null
rm -rf .tmp/ 2>/dev/null

# Remove build artifacts
echo "Removing build artifacts..."
rm -rf build/ 2>/dev/null
rm -rf dist/ 2>/dev/null
rm -rf *.egg-info 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Remove LaTeX auxiliary files (keep PDF)
echo "Removing LaTeX auxiliary files..."
cd docs 2>/dev/null
rm -f *.aux *.log *.out *.toc *.synctex.gz 2>/dev/null
cd "$PROJECT_ROOT"

# Remove unnecessary documentation files
echo "Removing redundant documentation..."
rm -f README_OLD.md 2>/dev/null
rm -f VERIFICATION_SUMMARY.txt 2>/dev/null
rm -f DOCUMENTATION_UPDATE_SUMMARY.md 2>/dev/null

# Remove test data and large unnecessary files
echo "Removing test data..."
rm -rf data/test/ 2>/dev/null
rm -rf data/temp/ 2>/dev/null

# Remove Node modules (if any)
echo "Removing Node modules..."
rm -rf node_modules/ 2>/dev/null

# Remove large DEM files (users should download separately)
echo "Checking large data files..."
if [ -d "data/dem" ]; then
    echo "  Note: DEM files in data/dem/ should be downloaded separately"
    echo "  Consider adding download script instead of committing large files"
fi

# Remove duplicate/old model checkpoints
echo "Cleaning model checkpoints..."
if [ -d "models/marl" ]; then
    find models/marl -name "*_old.pkl" -delete 2>/dev/null
    find models/marl -name "*_backup.pkl" -delete 2>/dev/null
fi

# Remove emoji removal scripts (task complete)
echo "Removing cleanup scripts..."
rm -f scripts/remove_emojis.py 2>/dev/null
rm -f scripts/remove_emojis_python.py 2>/dev/null

# Remove unnecessary status files
echo "Removing status tracking files..."
rm -f PHASE_COMPLETION_CHECKLIST.md 2>/dev/null
rm -f CHAT_APP_VERIFICATION.md 2>/dev/null
rm -f VISUAL_TESTING_GUIDE.md 2>/dev/null
rm -f TEST_RESULTS.md 2>/dev/null

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
env/
ENV/
.coverage
htmlcov/
*.log

# IDEs
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Project specific
data/cache/
data/temp/
logs/*.log
logs/*.jsonl
tmp/
temp/

# Models (too large for git)
models/vision/*.pth
models/vision/*.bin
data/dem/*.tif

# Environment
.env
*.key
*.pem

# Build
build/
dist/
*.egg-info/

# LaTeX
*.aux
*.log
*.out
*.toc
*.synctex.gz
EOF
fi

# Create .dockerignore if it doesn't exist
if [ ! -f ".dockerignore" ]; then
    echo "Creating .dockerignore..."
    cat > .dockerignore << 'EOF'
# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv/
env/

# IDEs
.vscode
.idea
*.swp

# Documentation (not needed in container)
docs/
*.md
!README.md

# Tests (not needed in production)
tests/
.pytest_cache/
.coverage
htmlcov/

# Logs (created at runtime)
logs/
*.log

# Temporary
tmp/
temp/
.DS_Store

# Large files
data/dem/*.tif
EOF
fi

echo ""
echo "Cleanup complete!"
echo ""
echo "Summary of actions:"
echo "  - Removed Python cache files"
echo "  - Removed virtual environments"
echo "  - Removed IDE/editor files"
echo "  - Removed log files"
echo "  - Removed build artifacts"
echo "  - Removed LaTeX auxiliary files"
echo "  - Created/updated .gitignore"
echo "  - Created/updated .dockerignore"
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Add essential files: git add ."
echo "  3. Commit: git commit -m 'Clean up for production'"
echo "  4. Add screenshots to docs/screenshots/"
echo "  5. Push to GitHub: git push"
echo ""
