## Project Structure

```
project/
│
├── data/
│   └── documents.pdf
│
├── vector_store/
│   └── (Generated vector / memory files stored here)
│
├── create_memory_with_llm.py
├── create_memory_for_llm.py
├── requirements.txt
└── README.md
```

---

## Vector / Memory Files

After running the memory creation script, the generated vector or memory files should be stored inside the **vector_store** folder.

Example:

```
vector_store/
│
├── index.pkl
├── memory.json
├── embeddings.db
```

This keeps the project organized and separates:

* Raw documents → `data`
* Generated memory/vector files → `vector_store`
* Processing scripts → root folder

---

## Usage Workflow

Step 1 — Create memory from documents:

```
python create_memory_with_llm.py  
```

Step 2 — Use the created memory for question answering:

```
python create_memory_for_llm.py
```

Make sure the generated memory/vector files exist inside the **vector_store** folder before running the query script.
      
