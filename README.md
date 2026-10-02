# Offline Legal Contract Agent

A local contract-analysis assistant for PDF and DOCX agreements. The app extracts
contract text, keeps its source locations, retrieves relevant passages with local
embeddings, and asks a local Ollama model to answer using only those passages.
It includes contract Q&A, a key-clause checklist, a summary, and attention points.

## How it works

1. Upload one PDF or DOCX in the Streamlit app. The file is held in the app session;
   it is not copied to a project folder or saved by this application.
2. Text is extracted locally. PDF text keeps its page number. DOCX text keeps its
   heading and paragraph or table-row location because page numbers depend on how
   the document is rendered.
3. Text is split into overlapping passages. Each passage keeps its source location.
4. Ollama creates an embedding for each passage using `nomic-embed-text`. When you
   ask a question or request an analysis, the app embeds that request and ranks the
   passages with NumPy cosine similarity.
5. Passages below the relevance threshold are discarded. If nothing relevant
   remains, a question is answered with a “not found” response without asking the
   chat model to generate an answer.
6. The remaining passages are sent to the local chat model (`qwen3:1.7b` by default).
   The interface displays the model's answer beside the retrieved passages, their
   source locations, and relevance scores so you can check the support in the
  original contract. Answers without valid references to the retrieved passages
  are withheld rather than shown as grounded answers.

The clause checklist searches payment, term and termination, confidentiality,
liability, indemnification, governing law, service levels, and penalties/remedies.
The summary and attention-point views use separate retrieval queries so they can
cover a range of contract topics. These are retrieval-based reviews, not guaranteed
complete legal analyses.

## Requirements

- Python 3.11 or later
- Ollama installed and running on this computer
- Local Ollama models `qwen3:1.7b` and `nomic-embed-text`
- Windows PowerShell commands below assume this project directory is the current
  directory

The configured models can be changed with environment variables. `qwen3:1.7b` is
the default because it is already installed for this project; `llama3.2` can be
selected later by changing `OLLAMA_CHAT_MODEL`.

## Setup

Create a virtual environment and install the declared packages from the official
Python Package Index:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip --index-url https://pypi.org/simple
python -m pip install -e ".[dev]" --index-url https://pypi.org/simple
```

If PowerShell blocks activation, run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" --index-url https://pypi.org/simple
```

Confirm that Ollama is running and both models are present:

```powershell
ollama list
```

If you still need to download a model, do so while connected to the internet, before
using the app offline:

```powershell
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
```

Optional local configuration can be set by copying `.env.example` to `.env` and
editing the values. `.env` is ignored by Git. The app rejects non-local Ollama
URLs so contract text is not sent to a remote model endpoint.

## Run

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. The server is configured to bind to
`127.0.0.1`; it is not exposed to your network. Upload a contract, then use the
tabs to ask questions, extract key clauses, create a summary, or flag provisions
for review. The relevance slider lets you make retrieval stricter or more
permissive; a higher threshold can reduce unsupported answers but may also miss
relevant passages.

## Test

An upload-ready, fictional DOCX contract is included at
[`examples/sample_contract.docx`](examples/sample_contract.docx). It is synthetic
software test data, not a real agreement or legal template. See
[`examples/README.md`](examples/README.md) for questions to try and expected facts.

```powershell
python -m pytest
```

The unit tests cover PDF page metadata, DOCX structural locations, chunk source
preservation, and abstention when retrieval finds no relevant passage. They do not
require a running Ollama server. For a local integration check, run the app with a
sample agreement and verify answers about payment, termination, SLA, liability,
indemnification, confidentiality, and governing law. Also ask about a term that is
not present and inspect the displayed evidence for every answer.

## Privacy and limitations

- Runtime document processing, embeddings, and chat use local software and models.
  Streamlit usage telemetry is disabled, and the server listens only on localhost.
- The app does not save uploaded contract contents or its vector index to disk.
  Content remains in memory for the browser session; close the session/app to clear
  it. Ollama manages its own local model files and logs.
- The default chat model is a small local model. It may miss nuance or produce
  incorrect analysis. A relevance threshold and evidence display reduce risk but
  cannot guarantee that every generated statement is correct; verify important
  conclusions against the original document.
- Retrieval can omit relevant passages, especially for very long or unusual
  contracts. A summary or clause result is based on retrieved passages, not a
  guarantee that every provision was found.
- Scanned/image-only PDFs are not OCR processed. Upload a text-selectable PDF.
- DOCX citations are heading/paragraph/table locations, not page numbers. PDF page
  citations refer to the PDF's own page numbering.
- This tool supports legal professionals with document review; it does not provide
  legal advice or replace professional judgment.
