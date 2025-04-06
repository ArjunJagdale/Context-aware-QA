---
title: Context Aware QA
emoji: 🔥
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
short_description: It uses the webpage as context to generate helpful answers
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
**access the tool here** - https://huggingface.co/spaces/ajnx014/Context-Aware-QA

# 📚 Context-Aware QA from a Webpage (FAISS + HuggingFace)

This project allows users to ask context-aware questions using the content of any **publicly accessible webpage**. It uses **FAISS** for efficient retrieval and **HuggingFace's Zephyr-7B** model to generate human-like answers based on context.

---

## 🚀 How It Works

1. Enter a webpage URL (like a blog, article, or documentation).
2. Click **"Load Webpage"** – this fetches and embeds the content using `bge-base-en-v1.5` embeddings.
3. Ask a question related to the webpage.
4. The model generates a response **strictly based on the context**.  
   If it doesn't know the answer, it'll say: _"I don't know."_

---

## 🧠 Tech Stack

- **Gradio** – for the interactive UI
- **LangChain** – for chaining components and context handling
- **HuggingFace Hub** – for Zephyr-7B LLM and embedding models
- **FAISS** – for vector similarity search

---

## 💡 What’s happening behind the scenes? It's powered by RAG (Retrieval-Augmented Generation)

Here’s how I implemented RAG in this project:

**Retrieval** 📚
Using FAISS and BAAI/bge-base-en-v1.5 embeddings, I split the webpage into chunks and retrieve the top relevant ones based on the user's question.

**Augmentation** 🧩
The retrieved chunks are passed as "context" into the prompt — this ensures the LLM stays grounded in factual, source-based information.

**Generation** 🤖
Using zephyr-7b-alpha from Hugging Face, the model answers based only on the provided context. If no answer is found, it says: "I don't know." — no guessing!

## ⚠️ Known Limitations
- Only works with public webpages (not behind logins or paywalls)
- Cannot process PDFs, images, or JavaScript-heavy content
- Limited understanding of math formulas or code-heavy content

## 💡 Example Use Cases
- Ask questions from blog posts or documentation
- Extract info from a product or research article
- Generate context-aware summaries
