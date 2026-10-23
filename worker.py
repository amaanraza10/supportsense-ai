"""Runs ONE model job in its own process, so every byte of memory goes back to the server when it exits.
Usage: python worker.py job.json out.json   (called by app.py, not meant to be run by hand)."""
import json
import sys


def classify(job):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo = job["repo"]
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo, low_cpu_mem_usage=True).eval()

    # Label names from the model config; if it only has LABEL_0, LABEL_1, ... fall back to the
    # alphabetical class list from the data (the usual LabelEncoder order).
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    classes = job["classes"]
    generic = all(str(v).upper().startswith("LABEL_") for v in id2label.values())
    guessed = bool(generic and len(classes) == len(id2label))
    names = {i: classes[i] for i in range(len(classes))} if guessed else id2label

    texts, batch, labels = job["texts"], job["batch"], []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        with torch.no_grad():
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
            probs = torch.softmax(model(**enc).logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        labels += [[names[k], float(c)] for c, k in zip(conf.tolist(), idx.tolist())]
    return {"labels": labels, "guessed": guessed}


def embed(job):
    import torch
    from sentence_transformers import SentenceTransformer

    try:
        m = SentenceTransformer("sentence-transformers/LaBSE", device="cpu",
                                model_kwargs={"torch_dtype": torch.bfloat16})   # half the memory
    except Exception:
        m = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
    v = m.encode(job["texts"], batch_size=job["batch"], convert_to_tensor=True, normalize_embeddings=True)
    return {"vectors": v.float().cpu().numpy().tolist()}


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        job = json.load(f)
    result = classify(job) if job["kind"] == "classify" else embed(job)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(result, f)
