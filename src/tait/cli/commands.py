"""CLI command implementations. Heavy ML imports stay inside commands."""
import hashlib
import math
import os
import pickle
import time
from pathlib import Path

from .ui import progress_bar


def cmd_version(_args):
    from ..version import __version__
    print(f"TAIT {__version__}")


def cmd_doctor(_args):
    from ..hardware.doctor import run_doctor
    run_doctor()


def cmd_benchmark(args):
    from ..hardware.benchmark import run_benchmark
    run_benchmark(**{k: getattr(args, k) for k in ("d_model", "d_ff", "batch_size", "seq_len", "dataset_size", "epochs")})


def _build_batch_index(order_by_len, batch_size, rng):
    batches = [order_by_len[i:i + batch_size] for i in range(0, len(order_by_len), batch_size)]
    idx = list(range(len(batches))); rng.shuffle(idx)
    return [batches[i] for i in idx]


def _batch_from_indices(sequences_ids, batch_idx, pad_id, block_size):
    import numpy as np
    seqs = [sequences_ids[i] for i in batch_idx]
    maxlen = min(max(len(s) for s in seqs), block_size + 1)
    x = np.full((len(seqs), maxlen - 1), pad_id, dtype=np.int64)
    y = np.full((len(seqs), maxlen - 1), pad_id, dtype=np.int64)
    valid = np.zeros((len(seqs), maxlen - 1), dtype=bool)
    tgt_valid = np.zeros((len(seqs), maxlen - 1), dtype=bool)
    for row, s in enumerate(seqs):
        s = s[:maxlen]
        xi, yi = s[:-1], s[1:]
        L = len(xi)
        x[row, :L], y[row, :L] = xi, yi
        valid[row, :L] = True; tgt_valid[row, :L] = True
    return x, y, valid, tgt_valid


def cmd_train(args):
    import numpy as np
    from ..core.model import MiniGPT
    from ..core.tokenizer import BPETokenizer, SEP, END
    from ..data.jsonl import load_jsonl
    from ..hardware.termux import wake_lock_acquire, wake_lock_release

    if not args.data:
        raise SystemExit("Training requires --data or a config file containing data.")
    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}")
    if not args.no_wake_lock:
        wake_lock_acquire()
    try:
        print(f"[1/5] Loading dataset: {args.data}")
        all_data, invalid = load_jsonl(args.data)
        if not all_data:
            raise SystemExit("Dataset contains 0 valid examples. Expected JSONL rows with prompt/response (or aliases).")
        print(f"  Loaded {len(all_data)} examples; skipped {invalid} invalid/empty rows.")

        print("[2/5] Training/loading BPE tokenizer...")
        training_texts = [d["prompt"] for d in all_data] + [d["response"] for d in all_data]
        data_hash = hashlib.sha256(("\x1f".join(training_texts) + f"|{args.vocab_size}|{args.bpe_sample_size}").encode("utf-8")).hexdigest()[:16]
        cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"bpe_cache_{data_hash}.pkl"
        tokenizer = BPETokenizer(); t0 = time.time()
        if cache_path.exists():
            state = pickle.loads(cache_path.read_bytes())
            tokenizer.merges = state["merges"]; tokenizer.vocab = state["vocab"]
            tokenizer.sep_id, tokenizer.end_id, tokenizer.pad_id = state["sep_id"], state["end_id"], state["pad_id"]
            print(f"  Loaded cache in {time.time()-t0:.2f}s: {cache_path}")
        else:
            tokenizer.train(training_texts, vocab_size=args.vocab_size, sample_size=args.bpe_sample_size)
            cache_path.write_bytes(pickle.dumps({"merges": tokenizer.merges, "vocab": tokenizer.vocab,
                "sep_id": tokenizer.sep_id, "end_id": tokenizer.end_id, "pad_id": tokenizer.pad_id}))
            print(f"  Trained in {time.time()-t0:.1f}s; cache: {cache_path}")

        print("[3/5] Encoding dataset...")
        encoded_sequences = []
        sequences = [d["prompt"] + SEP + d["response"] + END for d in all_data]
        for i, s in enumerate(sequences):
            encoded_sequences.append(tokenizer.encode(s))
            if i % max(1, len(sequences)//50) == 0 or i == len(sequences)-1:
                progress_bar(i+1, len(sequences), "Encoding", newline_at_end=(i == len(sequences)-1))
        if args.max_seq_len:
            encoded_sequences = [ids[:args.max_seq_len] for ids in encoded_sequences]
            print(f"  Applied max sequence length: {args.max_seq_len}")
        lens = np.array([len(s) for s in encoded_sequences])
        block_size = int(lens.max())
        print(f"  Token lengths: min={lens.min()} p50={int(np.percentile(lens,50))} p95={int(np.percentile(lens,95))} max={lens.max()}")
        print(f"  Vocab={tokenizer.vocab_size_total} | block_size={block_size}")
        if block_size < 2:
            raise SystemExit("Encoded sequences are too short to train.")

        print("[4/5] Training model...")
        model = MiniGPT(tokenizer.vocab_size_total, pad_id=tokenizer.pad_id, d_model=args.d_model,
                        d_ff=args.d_ff, block_size=block_size)
        order_by_len = sorted(range(len(encoded_sequences)), key=lambda i: len(encoded_sequences[i]))
        rng_train = np.random.default_rng(0)
        epoch_times = []
        for epoch in range(args.epochs):
            t_epoch = time.time(); total_loss = 0.0; n_batches = 0
            batches = _build_batch_index(order_by_len, args.batch_size, rng_train)
            for bi, batch_idx in enumerate(batches):
                x, y, valid, tgt_valid = _batch_from_indices(encoded_sequences, batch_idx, tokenizer.pad_id, block_size)
                probs, cache = model.forward(x, valid)
                p = np.clip(probs, 1e-9, 1.0)
                ll = np.log(np.take_along_axis(p, np.clip(y,0,tokenizer.vocab_size_total-1)[...,None], axis=-1)[...,0])
                loss = -(ll * tgt_valid).sum()/max(tgt_valid.sum(),1)
                grads = model.backward(cache, y, tgt_valid); model.step(grads, lr=args.lr)
                total_loss += loss; n_batches += 1
                if bi % max(1,len(batches)//20) == 0 or bi == len(batches)-1:
                    progress_bar(bi+1,len(batches),f"Epoch {epoch+1}/{args.epochs}",newline_at_end=(bi==len(batches)-1))
            avg_loss = total_loss/max(n_batches,1); dt=time.time()-t_epoch; epoch_times.append(dt)
            recent = epoch_times[-5:]; remaining = (sum(recent)/len(recent))*(args.epochs-epoch-1)
            print(f"  loss={avg_loss:.4f} | {dt:.1f}s/epoch | ETA {remaining/60:.1f} min")
            if args.checkpoint_every and (epoch+1)%args.checkpoint_every==0:
                Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                model.save(args.checkpoint_path, tokenizer)
                print(f"  [checkpoint] {args.checkpoint_path}")

        print(f"Training complete. Final loss: {avg_loss:.4f}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output, tokenizer)
        print(f"[5/5] Saved model: {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")
    finally:
        if not args.no_wake_lock:
            wake_lock_release()


def cmd_chat(args):
    from ..core.model import MiniGPT
    from ..core.generation import generate_reply
    model, tokenizer = MiniGPT.load(args.model)
    if args.web:
        from ..web.server import run_server
        run_server(model, tokenizer, host=args.host, port=args.port, temperature=args.temperature,
                   top_k=args.top_k, top_p=args.top_p, max_tokens=args.max_tokens, open_browser=not args.no_browser)
        return
    print(f"Loaded {args.model} | vocab={tokenizer.vocab_size_total} block_size={model.block_size}")
    print(f"temperature={args.temperature} top_k={args.top_k} top_p={args.top_p}")
    print("Type 'exit' to quit\n")
    while True:
        try: msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt): print("\nBye!"); break
        if not msg: continue
        if msg.lower() in ("exit", "quit"): print("Bye!"); break
        print(f"Model: {generate_reply(model, tokenizer, msg, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)}\n")


def cmd_data_validate(args):
    from ..data.jsonl import load_jsonl
    rows, invalid = load_jsonl(args.data)
    print(f"Valid examples: {len(rows)}")
    print(f"Invalid/empty rows: {invalid}")
    if invalid and args.strict:
        raise SystemExit(1)


def cmd_data_inspect(args):
    from ..data.jsonl import load_jsonl
    rows, invalid = load_jsonl(args.data)
    print(f"Dataset: {args.data}")
    print(f"Valid: {len(rows)} | Invalid/empty: {invalid}")
    for i, row in enumerate(rows[:args.n], 1):
        print(f"\n--- Example {i} ---\nPrompt: {row['prompt']}\nResponse: {row['response']}")
