___

## STEP 1: Set up environment with `uv` 

### - STEP 1.1: If you want to build a new project folder. Follow the the steps.
     1.1.1 > `uv init`  - The uv init command is used to initialize a new project environment
     1.1.2 > `uv add openai` - This is create the .venv folder if not present and install the openai module

## OR
### - STEP 1.2: If you are cloning the repo, run the following command to set up
    - 1.2.1 > 'uv sync'


### 🌐 Related documentation:
    - https://www.anthropic.com/engineering/building-effective-agents

## ENV file details.
    - OPENAI_API_KEY=XXXXX
    - MODEL_NAME=XXXXX
    - HF_TOKEN=XXXXX    
    - LANGSMITH_TRACING=true
    - LANGSMITH_ENDPOINT=https://api.smith.langchain.com
    - LANGSMITH_API_KEY=
    - LANGSMITH_PROJECT=
    - SERPAPI_API_KEY=



### ⚙️ Setup of SerpAPI.
- https://serpapi.com/