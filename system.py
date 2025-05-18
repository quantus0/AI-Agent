import asyncio
from together import Together
from langchain.agents import AgentExecutor, create_react_agent
from langchain.llms import VLLM
from playwright.sync_api import sync_playwright
from restrictedpython import compile_restricted
import gradio as gr
from fastapi import FastAPI
import uvicorn

# Initialze
together_client = Together(api_key="your_together_api_key")

# Load the fine-tuned model 
llm = VLLM(model="your-account/Qwen2.5-14B-Instruct:ft-job-id", api_key="your_together_api_key")

# Web browsing tool using Playwright
def search_web(query):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"https://www.google.com/search?q={query}")
        results = page.query_selector_all("h3")
        return [r.inner_text() for r in results[:5]]

# Safe code execution tool using RestrictedPython
def execute_code(code):
    globals_dict = {}
    try:
        compiled_code = compile_restricted(code, '<string>', 'exec')
        exec(compiled_code, globals_dict)
        return globals_dict.get('result', 'No result')
    except Exception as e:
        return f"Error: {str(e)}"

# Tools for the agent
tools = [
    Tool(name="WebSearch", func=search_web, description="Search the web for information"),
    Tool(name="CodeExec", func=execute_code, description="Execute Python code")
]

# Multi-agent system
agent = create_react_agent(llm, tools)
executor = AgentExecutor(agent=agent, tools=tools)

# FastAPI backend
app = FastAPI()

@app.post("/agent")
async def run_agent(query: str):
    return {"result": executor.run(query)}

# Gradio frontend
def chat_with_agent(user_input):
    return executor.run(user_input)

interface = gr.Interface(
    fn=chat_with_agent,
    inputs="text",
    outputs="text",
    title="Custom AI Agent"
)

# Run both FastAPI and Gradio
async def main():
    # Start Gradio interface in a non-blocking way
    interface.launch(share=True, prevent_thread_lock=True)
    
    # Start FastAPI server
    config = uvicorn.Config(app=app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
