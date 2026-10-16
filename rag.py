import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(Path(__file__).parent / ".env")

DOCS_DIR = Path(__file__).parent / "docs"
MODEL = "gemini-3.5-flash"

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(DOCS_DIR.glob("*.txt"))}
vectorizer = TfidfVectorizer(stop_words="english")
doc_matrix = vectorizer.fit_transform(docs.values())


def retrieve(question: str, k: int = 2) -> list[str]:
    scores = cosine_similarity(vectorizer.transform([question]), doc_matrix)[0]
    texts = list(docs.values())
    return [texts[i] for i in scores.argsort()[::-1][:k] if scores[i] > 0]


def answer_with_context(question: str, model: str = MODEL) -> tuple[str, list[str]]:
    chunks = retrieve(question)
    context = "\n\n".join(chunks) or "(no relevant documents)"
    resp = client.models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction="Answer using ONLY the context below. If the answer is not in the context, say \"I don't know.\"\n\nContext:\n" + context,
        ),
    )
    return resp.text or "", chunks


def answer(question: str, model: str = MODEL) -> str:
    return answer_with_context(question, model)[0]


if __name__ == "__main__":
    print(answer(" ".join(sys.argv[1:])))
