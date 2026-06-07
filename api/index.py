import os
from flask import Flask, request, jsonify
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

app = Flask(__name__)


CHUNK_SIZE    = 512
OVERLAP_RATIO = 0.2
TOP_K         = 10

EMBED_MODEL   = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL    = "4UHRUIN-gpt-5-mini"
LLMOD_BASE    = "https://api.llmod.ai/v1"

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open internet, "
    "or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    "\"I don't know based on the provided Medium articles data.\" "
    "Always explain your answer using the given context, quoting or paraphrasing "
    "the relevant article passage or metadata when helpful."
)


def _get_oai() -> OpenAI:
    return OpenAI(api_key=os.environ["LLMOD_API_KEY"], base_url=LLMOD_BASE)


def _get_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.index(os.environ["PINECONE_INDEX_NAME"])


@app.route("/api/prompt", methods=["POST"])
def prompt():
    body = request.get_json(force=True, silent=True) or {}
    question = str(body.get("question", "")).strip()

    if not question:
        return jsonify({"error": "question is required"}), 400

    oai   = _get_oai()
    index = _get_index()

    emb_resp = oai.embeddings.create(model=EMBED_MODEL, input=question)
    query_vec = emb_resp.data[0].embedding

    result = index.query(vector=query_vec, top_k=TOP_K, include_metadata=True)

    matches = result.matches or []
    context = [
        {
            "article_id": str(m.metadata.get("article_id", "")),
            "title":      str(m.metadata.get("title", "")),
            "authors":    str(m.metadata.get("authors", "")),
            "chunk":      str(m.metadata.get("chunk_text", "")),
            "score":      float(m.score),
        }
        for m in matches
    ]


    context_block = "\n\n---\n\n".join(
        f'[{i + 1}] Title: "{c["title"]}"\nAuthors: {c["authors"] or "unknown"}\nPassage: {c["chunk"]}'
        for i, c in enumerate(context)
    )

    context_out = [
        {k: v for k, v in c.items() if k != "authors"}
        for c in context
    ]
    user_message = (
        f"Context from Medium articles:\n\n{context_block}\n\n"
        f"Question: {question}"
    )

    augmented_prompt = {"System": SYSTEM_PROMPT, "User": user_message}

    chat_resp = oai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )
    response_text = chat_resp.choices[0].message.content or ""

    return jsonify({
        "response":        response_text,
        "context":         context_out,
        "Augmented_prompt": augmented_prompt,
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "chunk_size":    CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k":         TOP_K,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
