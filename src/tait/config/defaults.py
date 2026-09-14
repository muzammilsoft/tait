DEFAULTS = {
    "train": {
        "data": "examples/datasets/mixed.jsonl", "output": "tait_model.npz", "epochs": 10,
        "batch_size": 16, "lr": 0.003, "d_model": 32, "d_ff": 64, "vocab_size": 1000,
        "bpe_sample_size": 3000, "max_seq_len": None, "checkpoint_every": 10,
        "checkpoint_path": "checkpoint_latest.npz", "best_checkpoint_path": "checkpoint_best.npz",
        "cache_dir": ".tait-cache", "no_wake_lock": False, "validation_split": 0.1,
        "patience": 10, "min_delta": 0.001, "early_stopping": True, "reasoning": True,
        "response_only_loss": True, "seed": 42,
    },
    "benchmark": {"d_model": 32, "d_ff": 64, "batch_size": 64, "seq_len": 64, "dataset_size": 5000, "epochs": 60},
    "chat": {"temperature": 0.7, "top_k": 8, "top_p": 0.9, "max_tokens": 100, "host": "127.0.0.1", "port": 7860, "show_reasoning": False},
}
