# app.py

import os
import re
import gradio as gr
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🔐 Load token from Hugging Face Repo Secret
HF_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Setup embeddings and LLM
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_token,
    model_name="BAAI/bge-base-en-v1.5"
)

model = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

# Prompt template
system_prompt = """You are an AI Assistant. Only answer questions using the context provided.
If the answer is not in the context, say "I don't know." Strictly do not use prior knowledge."""

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=f"""
<|system|>>
{system_prompt}
</s>
<|user|>
Context:
{{context}}

Question:
{{query}}
</s>
<|assistant|>
"""
)

llm_chain = LLMChain(prompt=prompt_template, llm=model, verbose=False)

# Global vectorstore
vectorstore = None

# Step 1: Load and embed webpage
def load_webpage_and_prepare(url):
    global vectorstore
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
        chunked_docs = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunked_docs, embeddings)
        return "✅ Web page loaded successfully! You can now ask questions."
    except Exception as e:
        return f"❌ Error loading the webpage: {e}"

# Step 2: Ask a question
def get_answer(query):
    global vectorstore
    if vectorstore is None:
        return "⚠️ Please load a webpage first."

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(query)
    relevant_context = "\n".join([doc.page_content for doc in retrieved_docs])

    if not relevant_context.strip():
        return "I don't know."

    result = llm_chain.invoke({"context": relevant_context, "query": query})
    raw_output = result.get("text", "").strip()

    if "</s>" in raw_output:
        final_answer = raw_output.split("</s>")[-1].strip()
    else:
        final_answer = raw_output

    final_answer = re.sub(r"<\|.*?\|>", "", final_answer).strip()
    return final_answer or "I don't know."

# Gradio UI
with gr.Blocks(theme=gr.themes.Base(), css="""
    .scroll-box { max-height: 300px; overflow: auto; }
    .gr-accordion-label { font-weight: bold !important; font-size: 1.1rem !important; }
""") as demo:

    # Title outside instructions
    gr.Markdown("## 📚 Context-Aware QA from a Webpage (FAISS + HuggingFace) powered by RAG (Retrieval-Augmented Generation)")

    # Collapsible Instructions
    with gr.Accordion("ℹ️ How to Use This Tool (Click to Expand)", open=False):
        gr.Markdown(
            """
            This tool allows you to ask questions based on the content of **any public webpage**.
            Instead of giving exact snippets from the page, it **uses the webpage content as context** to generate helpful, human-like answers.

            ### 🚀 Steps to Use
            1. Enter a webpage URL (like a blog or documentation).
            2. Click **"Load Webpage"** – it fetches and processes the page in the background.
            3. Enter your question and press **"Submit Question"** (or just hit Enter).

            ### ⚠️ Known Limitations
            - It doesn't handle **math formulas, LaTeX, or code-heavy pages** very well.
            - Only works with **publicly accessible text-based pages** (no PDFs or login-based content).

            ---
            ✅ If the answer can't be found from the content, it will say:
            _"I don't know."_
            """
        )

    # URL input
    url_input = gr.Textbox(label="🌐 Enter Webpage URL", placeholder="https://example.com")
    load_button = gr.Button("Load Webpage")
    load_output = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("---")

    # Question input
    question_input = gr.Textbox(label="❓ Ask a Question", lines=2, placeholder="Ask a question from the loaded webpage...")
    submit_button = gr.Button("Submit Question")
    answer_output = gr.Textbox(label="📤 Answer", lines=6, elem_classes=["scroll-box"])

    # Button actions
    load_button.click(fn=load_webpage_and_prepare, inputs=url_input, outputs=load_output)
    submit_button.click(fn=get_answer, inputs=question_input, outputs=answer_output)
    question_input.submit(fn=get_answer, inputs=question_input, outputs=answer_output)

demo.launch()
