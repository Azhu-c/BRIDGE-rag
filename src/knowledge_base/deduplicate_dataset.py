import json
import sqlite3
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def init_sqlite(sqlite_db: str):
    conn = sqlite3.connect(sqlite_db)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS processed (hash TEXT PRIMARY KEY)""")
    conn.commit()
    return conn


def get_minhash(text: str, num_perm: int = 128):
    m = MinHash(num_perm=num_perm)
    for word in set(text.split()):
        m.update(word.encode("utf8"))
    return m


def deduplicate_dataset(input_file: str, output_file: str, sqlite_db: str, ir_bb_max_len: int = 10000, similarity_threshold: float = 0.95, num_perm: int = 128):
    conn = init_sqlite(sqlite_db)
    c = conn.cursor()
    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
    index = 0

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []
    for entry in tqdm(data, desc="Deduplicating"):
        if not all(str(entry.get(k, "")).strip() for k in ["ir_bb", "asm_bb"]):
            continue
        if len(entry["ir_bb"]) > ir_bb_max_len:
            continue

        asm_text = entry["asm_bb"]
        text_hash = str(hash(asm_text))
        c.execute("SELECT 1 FROM processed WHERE hash=?", (text_hash,))
        if c.fetchone():
            continue

        mh = get_minhash(asm_text, num_perm)
        if not lsh.query(mh):
            lsh.insert(f"m{index}", mh)
            cleaned_data.append(entry)
            c.execute("INSERT INTO processed (hash) VALUES (?)", (text_hash,))
            index += 1

    conn.commit()
    conn.close()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(cleaned_data)} deduplicated records to {output_file}")
    return cleaned_data
