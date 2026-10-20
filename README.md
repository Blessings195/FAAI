# LearnPath AI: A Multi-Agent System for Personalized Learning and Knowledge Gap Remediation

LearnPath AI is a multi-agent system designed to support personalized learning by identifying knowledge gaps and helping learners address them.

This repository contains the source code and project report for LearnPath AI. All application code is located in the `FAAI_CA3/learnpath-ai/learnpath` directory.

## Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

* Python 3.10 or newer
* Git
* VS Code or another code editor (optional)

### Installation and Setup

**1. Clone the repository**

```bash
git clone <repository-url>
cd <repository-folder>
```

Replace `<repository-url>` with your repository's Git URL and `<repository-folder>` with the cloned directory's name.

**2. Create and activate a virtual environment**

Create the environment:

```bash
python -m venv .venv
```

Activate it:

* **Windows (Command Prompt):**

  ```bash
  .venv\Scripts\activate
  ```

* **Windows (PowerShell):**

  ```powershell
  .venv\Scripts\Activate.ps1
  ```

* **Linux/macOS:**

  ```bash
  source .venv/bin/activate
  ```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch the application**

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, typically `http://localhost:8501`, to access the application.

## Running the Tests

To run the evaluation module without launching the Streamlit interface, execute:

```bash
python -m learnpath.evaluation
```

## Optional: Running with Ollama

LearnPath AI can optionally use a locally hosted language model through Ollama.

1. Install Ollama.

2. Download the Llama 3.1 model:

   ```bash
   ollama pull llama3.1
   ```

3. Install the Ollama integration:

   ```bash
   pip install langchain-ollama
   ```

4. Launch the application and select **Ollama** in the sidebar.

5. Click **Check Ollama Connection** to verify the connection.

Ensure that Ollama is running and the selected model is available locally before using this option.
