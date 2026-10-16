import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rag


def call_api(prompt, options, context):
    query = context["vars"]["query"]
    model = options.get("config", {}).get("model", rag.MODEL)
    text, chunks = rag.answer_with_context(query, model)
    # metadata.context lets promptfoo's context-faithfulness check see what was retrieved
    return {"output": text, "metadata": {"context": "\n\n".join(chunks) or "(no relevant documents)"}}
