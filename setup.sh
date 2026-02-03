set -e

echo "Installing spaCy dependencies..."
uv run python -m spacy download en_core_web_sm
