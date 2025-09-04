export PYTHONPATH=/opt/render/project/src:$PYTHONPATH
#!/usr/bin/env bash
# start_server.sh
# Simple start script for deploying locally or on Render.
# Usage: ./start_server.sh
# Make sure DATABASE_URL is set in environment (Supabase connection string).
set -euo pipefail

# Optional: load .env file if present (requires `dotenv` or envsetup)
if [ -f .env ]; then
  echo "Loading .env into environment"
  set -a
  source .env
  set +a
fi

# ensure python deps installed (optional)
# pip install -r requirements.txt

# Run streamlit app
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_LOG_LEVEL=${STREAMLIT_LOG_LEVEL:-info}
echo "Starting Streamlit app dashboard/app.py"
exec streamlit run dashboard/app.py
