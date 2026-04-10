import subprocess
import sys
import shutil

# --- Setup: ensure ollama is installed and model is available ---

def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)

if not shutil.which("ollama"):
    print("Installing Ollama...")
    run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)

# Start ollama server in background if not already running
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try:
    run(["pip", "install", "-q", "langchain-ollama"], stdout=subprocess.DEVNULL)
except subprocess.CalledProcessError:
    run([sys.executable, "-m", "pip", "install", "-q", "langchain-ollama"])

model_id = "llama3.2:3b"

# Pull model only if not already present
result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
if model_id not in result.stdout:
    print(f"Pulling model {model_id} (this may take a few minutes)...")
    run(["ollama", "pull", model_id])

# --- LLM inference ---

from langchain_ollama import OllamaLLM

llm = OllamaLLM(model=model_id, temperature=0.5, num_predict=256)

query = input("Please enter your query: ")
print(llm.invoke(query))
