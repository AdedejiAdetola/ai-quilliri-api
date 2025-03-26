# AI Medical Diagnostics System -  AI QUILLIRI

## Overview

This project is an AI medical diagnostics system leveraging OpenAI, LangChain, and ChromaDB. The system downloads medical articles, processes them into structured documents, indexes them using embeddings, and allows users to query the indexed data for relevant information.

## Features

- **Web Scraping & Data Collection**: Downloads medical documents from various health-related websites.
- **File Processing**: Detects and renames HTML files using `python-magic`.
- **Document Indexing**: Uses `langchain` to parse, clean, and split documents into chunks.
- **Vector Storage**: Stores document embeddings using ChromaDB.
- **Querying System**: Implements similarity search for retrieving relevant medical content.
- **Natural Language Processing**: Uses OpenAI's `ChatOpenAI` for intelligent query responses.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install openai==0.27.1 langchain==0.0.191 chromadb==0.3.26 tiktoken==0.4.0
pip install python-magic dotenv tqdm pandas
```

## Usage

### 1. Environment Setup

Create a `.env` file and set your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Running the Script

Execute the main script to process and query medical documents:

```bash
python main.py
```

### 3. Querying the Data

After processing documents, you can query the database:

```python
query = "What is anxiety?"
results = db.similarity_search(query)
```

## File Structure

```
.
├── main.py                # Main script for processing and querying
├── requirements.txt       # List of dependencies
├── .env                   # Environment variables
├── data/                  # Directory containing downloaded documents
├── README.md              # This README file
```

## Future Enhancements

- Implement more advanced document cleaning and preprocessing.
- Improve retrieval accuracy using more sophisticated embedding models.
- Expand the dataset with additional medical resources.

## Contributors

- **Adetola Adedeji**

## License

This project is licensed under the MIT License.

