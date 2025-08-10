pip uninstall torch torchvision torchaudio
pip install uv 
source .venv/bin/activate
uv pip install -r requirements.txt

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash