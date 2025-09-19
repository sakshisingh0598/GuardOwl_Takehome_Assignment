# Scenario

At Guard Owl, supervisors often need quick answers to questions like:

- “What happened at Site S01 last night?”
- “Were there any geofence breaches at the west gate this week?”
- “Show me all incidents involving a red Toyota Camry.”

Your task is to prototype a lightweight version of our **Ask Guard Owl chatbox**: a service that ingests reports, retrieves relevant information, and produces a structured answer with citations.

### **Dataset Provided**

- File: `guard_owl_mock_reports.json` (52 mock security reports).
  [guard_owl_mock_reports.json]
- Schema:

```jsx
{
"id": "r123",
"siteId": "S12",
"date": "2025-08-01T02:00:00Z",
"guardId": "G45",
"text": "Guard observed a red Toyota Camry loitering near the west gate."
}
```

# Your Task

1. **Ingest Reports**
   - Load the JSON data.
   - Create embeddings for each report (using OpenAI, HuggingFace, or any library).
   - Store them in a vector index (FAISS, Pinecone, or MongoDB Atlas Vector Search).
2. **Implement Retrieval**
   - Given a query + optional filters (`siteId`, `dateRange`), retrieve the 3 most relevant reports.
   - Use both **semantic similarity** (via embeddings) and **keyword filtering**.
3. **Build an API** (FastAPI or Flask preferred)
   - `POST /query { query, siteId?, dateRange? }`
   - Returns:

```jsx
{
"answer": "Summary text here",
"sources": ["r123", "r456", "r789"]
}
```

- `answer`: a short summary of the results (use an LLM if you want, otherwise simple concatenation is fine).
- `sources`: IDs of the reports used.

1. **Documentation**

- Add a `README.md` that explains:
  - How to run your service locally.
  - Design choices you made.
  - How you would scale this to **1M+ reports**.
  - How you’d extend it to include **timesheets** or **geofence events**.

# Deliverables

- A GitHub repo (or zip file) containing:
  - Your code.
  - `README.md` with instructions and design notes.
- API should be runnable locally (no need to deploy).
- Timebox yourself to **3–4 hours**; we value clarity and reasoning over completeness.

### **Evaluation Criteria**

- **Execution (40%)** — Does retrieval work? Are relevant results returned?
- **Code Quality (25%)** — Is the code clean, modular, and documented?
- **Design Reasoning (25%)** — Are scaling and extension ideas explained well?
- **Communication (10%)** — Is the README clear and easy to follow?
