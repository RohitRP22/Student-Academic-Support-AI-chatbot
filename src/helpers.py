# helper.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

class StudentRAGHelper:
    def __init__(self, csv_path="D:\Projects\Student-Academic-Support-AI-chatbot\data\students.csv", model_name="all-MiniLM-L6-v2"):
        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Convert each row into a text chunk
        self.documents = self.df.apply(self.row_to_text, axis=1).tolist()

        # Embedding model
        self.model = SentenceTransformer(model_name)

        # Build FAISS index
        self.index, self.id_to_doc = self.build_index()

    def row_to_text(self, row):
        return (
            f"Student {row['Student Name']} studied {row['Subject']} "
            f"and scored {row['Marks']} marks. "
            f"They completed {row['Assignments Completed']} assignments, "
            f"with attendance {row['Attendance %']}%. "
            f"Remarks: {row['Remarks']}."
        )

    def build_index(self):
        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        id_to_doc = {i: doc for i, doc in enumerate(self.documents)}
        return index, id_to_doc

    def retrieve(self, query, top_k=3):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, top_k)
        return [self.id_to_doc[i] for i in I[0]]


