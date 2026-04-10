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
    run(["pip", "install", "-q", "langchain-ollama", "gradio"], stdout=subprocess.DEVNULL)
except subprocess.CalledProcessError:
    run([sys.executable, "-m", "pip", "install", "-q", "langchain-ollama", "gradio"])

model_id = "llama3.2:3b"

# Pull model only if not already present
result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
if model_id not in result.stdout:
    print(f"Pulling model {model_id} (this may take a few minutes)...")
    run(["ollama", "pull", model_id])

# --- LLM inference ---

import gradio as gr
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model=model_id, temperature=0.5, num_predict=512)

def generate_response(prompt_txt):
    return llm.invoke(prompt_txt)

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Local Ollama Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port=7860)
