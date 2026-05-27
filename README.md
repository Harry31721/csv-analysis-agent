<div align="center">

# 🤖 CSV Analysis Agent

### An agentic AI that reads, analyses, and fixes your data — autonomously.

Upload any CSV. Ask a question in plain English.  
The agent reasons step-by-step, writes real Python code, executes it, and returns charts and insights.

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=for-the-badge&logo=groq&logoColor=white)

</div>

---

## 📌 What It Does

Most AI demos are chatbots that *talk*. This one **acts**.

You upload a messy CSV and ask something like *"find all data quality issues and write code to fix them"*. The agent:

1. **Reads** the file to understand its structure
2. **Profiles** it to detect nulls, duplicates, type errors, and outliers
3. **Writes Python code** to investigate or fix the problem
4. **Executes that code** and returns the output — including charts
5. **Explains its findings** in plain English

All of this happens autonomously in a multi-step reasoning loop powered by [LangGraph](https://github.com/langchain-ai/langgraph).

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Data Profiling** | Automatically detects nulls, duplicates, type mismatches, inconsistent casing, mixed date formats |
| 📊 **Visualisation** | Generates matplotlib / seaborn charts that match the dark UI |
| 🐍 **Code Generation** | Writes clean, runnable pandas code — not pseudocode |
| ⚙️ **Code Execution** | Actually runs the code and shows you real output |
| 🔗 **Multi-turn Memory** | Remembers the full conversation via LangGraph's `MemorySaver` |
| ⚡ **Fast Inference** | Groq's hardware acceleration makes Llama 3.3 70B feel instant |
| 🎨 **Professional UI** | Full dark theme, reasoning chain timeline, streaming indicators |

---

## 🏗️ Architecture

```
User Input (Streamlit)
        │
        ▼
┌───────────────────────────────────┐
│         LangGraph ReAct Loop      │
│                                   │
│   ┌─────────┐     ┌───────────┐   │
│   │  Agent  │────▶│   Tools   │   │
│   │  Node   │◀────│   Node    │   │
│   └────┬────┘     └───────────┘   │
│        │ (no more tool calls)     │
│        ▼                          │
│      END                          │
└───────────────────────────────────┘
        │
        ▼
  Final Answer + Charts
```

The agent node calls **Llama 3.3 70B via Groq** with three tools bound to it. The graph loops — agent decides → tools execute → agent sees results → decides again — until it has a complete answer.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Llama 3.3 70B via [Groq](https://console.groq.com) (free tier) |
| **Agent Framework** | [LangGraph](https://github.com/langchain-ai/langgraph) — ReAct loop with `MemorySaver` |
| **LLM Interface** | `langchain-groq` — `ChatGroq` with `bind_tools()` |
| **Tools** | `@tool` decorated Python functions via `langchain-core` |
| **Data** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn |
| **UI** | [Streamlit](https://streamlit.io) with custom dark CSS |
| **Code Execution** | `exec()` sandbox with stdout redirect + figure capture |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Harry31721/csv-analysis-agent.git
cd csv-analysis-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

```bash
cp .env.example .env
# Open .env and paste your key:
# GROQ_API_KEY=gsk_...
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
csv-analysis-agent/
│
├── app.py                        # Streamlit UI — chat, streaming, dark theme
│
├── agent/
│   ├── graph.py                  # LangGraph StateGraph (ReAct loop)
│   ├── state.py                  # AgentState TypedDict
│   └── tools.py                  # Three @tool functions
│
├── utils/
│   └── code_executor.py          # exec() sandbox + matplotlib figure capture
│
├── sample_data/
│   └── sales_data_messy.csv      # Demo dataset with 8 built-in data issues
│
├── .streamlit/
│   └── config.toml               # Dark theme (backgroundColor, primaryColor)
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🔧 The Three Tools

The agent autonomously decides which tools to call and in what order.

### `read_csv_file`
Previews the first N rows, column names, and data types. Used at the start of every analysis to understand the dataset's shape.

### `analyze_dataframe`
Returns a full statistical profile:
- Shape (rows × columns)
- Data types per column
- Missing value counts
- Duplicate row detection
- Descriptive statistics (mean, std, min/max)
- Value counts for categorical columns

### `execute_python_code`
The powerhouse. Accepts any Python code string, runs it in an isolated namespace where `df` is pre-loaded from the uploaded file, and captures:
- All `print()` output
- Any `matplotlib` / `seaborn` figures (returned as PNG and rendered inline)
- Full error tracebacks if the code fails

---

## 💬 Example Queries

Try these with the included sample dataset:

```
"What does this dataset look like?"
"Find all data quality issues"
"Show a bar chart of total sales by region"
"Detect outliers in the sales column using IQR"
"Write Python code to clean this data and standardise the date format"
"Which customer has the highest total spend?"
"Compare Widget A vs Widget B sales over time"
```

---

## 🗂️ Sample Dataset

`sample_data/sales_data_messy.csv` is a 30-row sales dataset deliberately packed with real-world data quality problems:

| Issue | Example |
|---|---|
| Mixed date formats | `2024-01-15`, `01/16/2024`, `15-01-2024` |
| Currency strings | `$1,200.00` instead of `1200.0` |
| Duplicate rows | Rows 3 and 4 are identical |
| Inconsistent casing | `North`, `north`, `NORTH` |
| Missing values | Blank `product` and `sales` cells |
| Negative values | `quantity = -5`, `sales = -$500` |
| Price outliers | Widget D at `$50,000` vs avg `$100` |

---

## 🧠 How the Agent Reasons

Below is a real trace for the query *"Find all data quality issues"*:

```
[Agent]  → Calling read_csv_file(file_path="...", n_rows=10)
[Tool]   ← 30 rows × 7 columns, columns: [date, product, sales, ...]

[Agent]  → Calling analyze_dataframe(file_path="...")
[Tool]   ← 2 missing values in 'product', 1 in 'sales'
           Duplicate rows: 2
           'sales' dtype: object (expected float)

[Agent]  → Calling execute_python_code(code="...", file_path="...")
           # Code checks for mixed date formats, negative values, outliers
[Tool]   ← OUTPUT: Found 3 date formats, 1 negative entry, 2 price outliers

[Agent]  ← Synthesises all tool results into a structured report
```

---

## 📸 Screenshots

> *Add screenshots or a demo GIF here after running the app.*

| Landing Page | Chat Interface | Reasoning Chain |
|---|---|---|
| *(screenshot)* | *(screenshot)* | *(screenshot)* |

---

## 🔒 Security Note

The `execute_python_code` tool runs arbitrary Python via `exec()`. This is intentional for a portfolio demo. For a production deployment, replace the executor with a sandboxed solution such as [E2B](https://e2b.dev) or a Docker-isolated subprocess.

---

## 📄 License

MIT — free to use, fork, and build on.

---

<div align="center">

Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [Groq](https://groq.com) · [Streamlit](https://streamlit.io)

⭐ Star this repo if you found it useful

</div>
