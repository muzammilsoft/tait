import argparse


def build_parser():
    p = argparse.ArgumentParser(prog="tait", description="TAIT - lightweight NumPy-first local AI training")
    p.add_argument("--version", action="store_true", help="show version")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("version", help="show version")
    sub.add_parser("doctor", help="environment and BLAS diagnostics")
    b = sub.add_parser("benchmark", help="speed test")
    for name, typ, default in (("d-model",int,None),("d-ff",int,None),("batch-size",int,None),("seq-len",int,None),("dataset-size",int,None),("epochs",int,None)):
        b.add_argument(f"--{name}", type=typ, default=default, dest=name.replace('-','_'))
    b.add_argument("--config", default=None)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("--config", default=None)
    t.add_argument("--data", default=None); t.add_argument("--output", default=None); t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch-size", type=int, default=None, dest="batch_size"); t.add_argument("--lr", type=float, default=None)
    t.add_argument("--d-model", type=int, default=None, dest="d_model"); t.add_argument("--d-ff", type=int, default=None, dest="d_ff")
    t.add_argument("--vocab-size", type=int, default=None, dest="vocab_size"); t.add_argument("--bpe-sample-size", type=int, default=None, dest="bpe_sample_size")
    t.add_argument("--max-seq-len", type=int, default=None, dest="max_seq_len"); t.add_argument("--checkpoint-every", type=int, default=None, dest="checkpoint_every")
    t.add_argument("--checkpoint-path", default=None, dest="checkpoint_path"); t.add_argument("--best-checkpoint-path", default=None, dest="best_checkpoint_path")
    t.add_argument("--cache-dir", default=None, dest="cache_dir")
    t.add_argument("--validation-split", type=float, default=None, dest="validation_split")
    t.add_argument("--patience", type=int, default=None)
    t.add_argument("--min-delta", type=float, default=None, dest="min_delta")
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=None)
    t.add_argument("--response-only-loss", action=argparse.BooleanOptionalAction, default=None, dest="response_only_loss")
    t.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=None, dest="early_stopping")
    t.add_argument("--no-wake-lock", action="store_true", default=None, dest="no_wake_lock")

    c = sub.add_parser("chat", help="chat with a model")
    c.add_argument("--model", required=True)
    c.add_argument("--web", action="store_true"); c.add_argument("--no-browser", action="store_true", dest="no_browser")
    c.add_argument("--host", default=None); c.add_argument("--port", type=int, default=None)
    c.add_argument("--temperature", type=float, default=None); c.add_argument("--top-k", type=int, default=None, dest="top_k")
    c.add_argument("--top-p", type=float, default=None, dest="top_p"); c.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    c.add_argument("--show-reasoning", action="store_true", default=None, dest="show_reasoning", help="show the generated reasoning trace")

    d = sub.add_parser("data", help="dataset utilities"); dsub = d.add_subparsers(dest="data_command", required=True)
    v = dsub.add_parser("validate"); v.add_argument("data"); v.add_argument("--strict", action="store_true")
    i = dsub.add_parser("inspect"); i.add_argument("data"); i.add_argument("-n", type=int, default=3)

    s = sub.add_parser("setup", help="first-run setup / repair"); s.add_argument("--repair", action="store_true"); s.add_argument("--yes", action="store_true")
    return p
