# Document Evidence Assistant

A local document question-answering prototype built to learn applied
AI engineering and retrieval-augmented generation (RAG).

## What it does

- Reads a fictional company travel policy.
- Retrieves relevant passages using semantic search.
- Generates answers locally through Ollama.
- Displays retrieved passages, similarity scores, and source citations.

## How it works

1. Split the document into non-empty lines.
2. Convert passages and the question into embeddings using MiniLM.
3. Rank passages using cosine similarity.
4. Send the top three passages and the question to Qwen.
5. Display the generated answer.

Both models are pretrained. This project does not train or fine-tune them.

## Technology

- Python
- Sentence Transformers: all-MiniLM-L6-v2
- Ollama: qwen3:4b-instruct-2507-q4_K_M

## Setup on Windows

Install Python 3.10 or later, Git, and Ollama.

Clone the repository:

```powershell
git clone https://github.com/akshaykvijayan/document-evidence-assistant.git
cd document-evidence-assistant
```

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

With Ollama running, download the answering model:

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Run the application:

```powershell
.\.venv\Scripts\python.exe semantic_search.py
```

Internet access is needed initially to download packages and models.
Model inference runs locally without a paid API key.

## Example questions

- What is the accommodation allowance?
- When must I submit receipts?
- Are personal holidays reimbursed?
- Are flights reimbursed for business trips?

## Initial manual evaluation

These are development observations from a small sample, not an accuracy benchmark.

| Question | Observation |
|---|---|
| Accommodation allowance | Retrieved the hotel passage, but the generated answer omitted “up to.” |
| Receipt deadline | Correctly answered within 30 days of completing the trip and cited the passage. |
| Personal holidays | Retrieved the correct passage, but generated a contradictory “Yes” followed by “not reimbursed.” |
| Flight reimbursement | Correctly reported that the document does not specify this. |

The prompt currently includes an accommodation/hotel wording example.
That question is therefore a development case, not an independent test.

## Limitations

- Supports one text document with line-based passages.
- Always retrieves up to three passages, even for unrelated questions.
- Similarity scores are not confidence percentages.
- Generated answers can omit conditions or contradict the source.
- Citations require verification.
- Prompt changes can affect behaviour.

## Next steps

- Organise the code into functions.
- Record a broader evaluation set.
- Support multiple documents.
- Improve answer reliability and source verification.

## Status

An early learning prototype under active development.