## Simple Search Engine V.1

This is a Python-based text search engine that uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to find the most relevant documents in a corpus. The project processes text files, builds a TF-IDF index, and allows users to perform keyword-based searches interactively.

# Feature
- Batch Document Processing: Automatically reads and processes all text files in a specified folder.
- TF-IDF Indexing: Converts documents into a vectorized form for efficient similarity computation.
- Cosine Similarity Matching: Ranks documents based on relevance to user queries.
- Interactive Query Interface: Allows users to input search queries and view results in real-time.

# Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.8+

Required Python libraries:
  - nltk
  - scikit-learn

# Setup and Usage
1. Clone or Download the Repository
- Download the project files to your local machine.

2. Prepare the Data
   make sure your text file is all in the ```SimpleText_auto``` folder
   make sure ```SimpleText_auto``` folder is in the same path with ```search_engine.py```

3. Run the Program
   Execute the program in cmd:
   ```
   python search_engine.py
   ```

# File Structure
```
project/
│
├── search_engine.py        # Main program file
├── /SimpleText_auto/       # Folder containing text files (data source)
└── README.md               # Project description and usage guide
```

# Project Workflow
1. Load Documents:
  - Reads all text files from the specified folder.

2. Preprocess Text:
  - Converts text to lowercase.
  - Removes punctuation and stopwords.
  - Tokenizes text into words.

3. Build TF-IDF Index:
  - Uses TfidfVectorizer to convert text into TF-IDF vectors.
  - Stores vectors for similarity computations.

4. Search Query:
  - Processes user input queries.
  - Calculates cosine similarity between the query and all documents.
  - Returns the top-N relevant documents.
