"""Score saved English answers; never regenerate answers or retrieval results."""
import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import string
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from power_eval.corpus import load_knowledge_corpus

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results/powergridqa_qwen_translated_en'
OUTPUT = ROOT / 'results/powergridqa_metrics_en'
METHODS = ['disco_full', 'without_rst', 'without_graph', 'without_planning', 'standard_rag']
CONFIG = dict(em='NFKC, lowercase, remove ASCII punctuation and English articles, collapse whitespace',
              rouge='rouge-score 0.1.2, rougeL F1, use_stemmer=False',
              bert='bert-score 0.3.13, roberta-large layer17, idf=False, rescale_with_baseline=False',
              summac='summac 0.0.4, SummaCZS, vitc, sentence granularity, default aggregation, max_doc_sents=100, max_input_length=500',
              art='original latency_seconds; reads the offline RST cache and includes graph/planning/generation; excludes offline cache construction and scoring')

def load(path):
    rows = {}
    if path.exists():
        for line in path.open():
            try:
                row = json.loads(line)
                rows[row['item_id']] = row
            except (ValueError, KeyError):
                continue
    return rows

def load_evidence_docs():
    """Load chunk texts using source IDs emitted by the retriever."""
    explicit = os.getenv('POWER_EVIDENCE_JSON', '')
    if explicit and Path(explicit).exists():
        return json.loads(Path(explicit).read_text(encoding='utf-8'))
    faiss_dir = os.getenv('POWER_FAISS_INDEX', '')
    if faiss_dir:
        evidence_json = Path(faiss_dir) / 'evidence_docs.json'
        if evidence_json.exists():
            return json.loads(evidence_json.read_text(encoding='utf-8'))
        docstore, _ = pickle.load((Path(faiss_dir) / 'index.pkl').open('rb'))
        return {f"{doc.metadata.get('source_file', '')}:{doc.metadata.get('chunk_id', 0)}": doc.page_content for doc in docstore._dict.values()}
    root = os.getenv('POWER_CORPUS_ROOT', 'data/corpus')
    # Debug-only fallback: the knowledge corpus is text, never the benchmark JSONL.
    return {document.source_id: document.text for document in load_knowledge_corpus(root)}

def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def token_f1(prediction, reference):
    pred = normalize(prediction).split()
    ref = normalize(reference).split()
    if not pred or not ref:
        return 1.0 if pred == ref else 0.0
    from collections import Counter
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)

def summary():
    result = {'configuration': CONFIG, 'methods': {}}
    for method in METHODS:
        original = load(INPUT / f'{method}.jsonl')
        merged = {key: {} for key in original}
        for stage in ['basic', 'bert', 'summac']:
            for key, row in load(OUTPUT / f'{method}.{stage}.jsonl').items():
                if key in merged:
                    merged[key].update(row)
        metrics = {}
        for field in ['token_f1', 'rouge_l_f1', 'bertscore_f1', 'summac', 'art_seconds']:
            values = [r[field] for r in merged.values() if r.get(field) is not None]
            metrics[field] = {'count': len(values), 'mean': sum(values)/len(values) if values else None}
        result['methods'][method] = {'expected': len(original), 'metrics': metrics}
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'summary.json').write_text(json.dumps(result, indent=2))
    with (OUTPUT / 'summary.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        fields = ['token_f1', 'rouge_l_f1', 'bertscore_f1', 'summac', 'art_seconds']
        writer.writerow(['method', 'N'] + fields + [key + '_count' for key in fields])
        for method, row in result['methods'].items():
            metrics = row['metrics']
            writer.writerow([method, row['expected']] + [metrics[k]['mean'] for k in fields] + [metrics[k]['count'] for k in fields])
    return result

def main():
    global INPUT, OUTPUT, METHODS
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['basic', 'bert', 'summac', 'summary'], required=True)
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--input-dir', type=Path, default=INPUT)
    ap.add_argument('--output-dir', type=Path, default=OUTPUT)
    ap.add_argument('--summac-mode', choices=['signed', 'entailment'], default='signed')
    args = ap.parse_args()
    INPUT, OUTPUT = args.input_dir, args.output_dir
    if (INPUT.name.startswith('powergridqa_ablation_') or
            (INPUT / 'disco_full.jsonl').exists()):
        METHODS = ['disco_full', 'without_rst', 'without_graph', 'without_planning', 'standard_rag']
        if args.methods == ['direct', 'standard_rag', 'graphrag', 'retrieve_plan', 'plan_retrieve', 'disco_rag']:
            args.methods = METHODS
    METHODS = args.methods
    CONFIG['summac_mode'] = args.summac_mode
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'summary':
        print(json.dumps(summary(), indent=2)); return
    scorer = None
    docs = None
    if args.stage == 'basic':
        from rouge_score.rouge_scorer import RougeScorer
        scorer = RougeScorer(['rougeL'], use_stemmer=False)
    elif args.stage == 'bert':
        from bert_score import BERTScorer
        scorer = BERTScorer(model_type='roberta-large', num_layers=17, lang='en',
                            idf=False, rescale_with_baseline=False, device=args.device, batch_size=8)
    else:
        from summac.model_summac import SummaCZS
        scorer = SummaCZS(granularity='sentence', model_name='vitc', device=args.device, use_cache=False,
                           use_ent=True, use_con=args.summac_mode == 'signed')
        docs = load_evidence_docs()
    for method in args.methods:
        source = load(INPUT / f'{method}.jsonl')
        dst = OUTPUT / f'{method}.{args.stage}.jsonl'
        done = load(dst)
        required_field = 'token_f1' if args.stage == 'basic' else ('bertscore_f1' if args.stage == 'bert' else 'summac')
        pending = [r for key, r in source.items() if key not in done or done[key].get(required_field) is None]
        if args.limit:
            pending = pending[:args.limit]
        with dst.open('a') as out:
            for start in range(0, len(pending), 8):
                batch = pending[start:start+8]
                if args.stage == 'bert':
                    p, r, f = scorer.score([x['answer'] for x in batch], [x['reference'] for x in batch])
                elif args.stage == 'summac':
                    evidence_batch = []
                    answer_batch = [x['answer'] for x in batch]
                    for row in batch:
                        ids = row.get('source_ids', [])
                        if method == 'direct':
                            evidence_batch.append('')
                        else:
                            if not ids or any(k not in docs for k in ids):
                                raise ValueError(f'Missing evidence: {method} {row["item_id"]}')
                            evidence_batch.append('\n\n'.join(docs[k] for k in ids))
                    summac_scores = scorer.score(evidence_batch, answer_batch, batch_size=len(batch))['scores']
                for j, row in enumerate(batch):
                    entry = {'item_id': row['item_id'], 'answer_sha256': hashlib.sha256(row['answer'].encode()).hexdigest()}
                    if args.stage == 'basic':
                        entry.update(token_f1=token_f1(row['answer'], row['reference']),
                            rouge_l_f1=scorer.score(row['reference'], row['answer'])['rougeL'].fmeasure,
                            art_seconds=row['latency_seconds'],
                            remaining_chinese=bool(re.search(r'[\u4e00-\u9fff]', row['answer'])))
                    elif args.stage == 'bert':
                        entry.update(bertscore_precision=float(p[j]), bertscore_recall=float(r[j]), bertscore_f1=float(f[j]))
                    else:
                        ids = row.get('source_ids', [])
                        if method == 'direct':
                            entry['summac'] = None
                        else:
                            entry['summac'] = float(summac_scores[j])
                    out.write(json.dumps(entry) + '\n')
                    out.flush()
                print(method, args.stage, start+len(batch), '/', len(pending), flush=True)
        print('COMPLETE', method, args.stage, flush=True)
    summary()

if __name__ == '__main__':
    main()
