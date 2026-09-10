import sys

from .parser import build_parser


def _merge(command, args):
    from ..config.loader import load_config, apply_cli
    cfg = load_config(getattr(args, "config", None), command)
    if command == "train":
        keys = ["data","output","epochs","batch_size","lr","d_model","d_ff","vocab_size","bpe_sample_size","max_seq_len","checkpoint_every","checkpoint_path","cache_dir","no_wake_lock"]
    elif command == "benchmark":
        keys = ["d_model","d_ff","batch_size","seq_len","dataset_size","epochs"]
    elif command == "chat":
        keys = ["host","port","temperature","top_k","top_p","max_tokens"]
    else:
        return args
    cfg = apply_cli(cfg, args, keys)
    for k,v in cfg.items(): setattr(args,k,v)
    return args


def main(argv=None):
    from ..setup.environment import is_first_run
    from ..setup.wizard import run_setup
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version or args.command == "version":
        from .commands import cmd_version; return cmd_version(args)
    if args.command is None:
        parser.print_help(); return 0
    if args.command == "setup":
        return run_setup(repair=args.repair, yes=args.yes)
    if is_first_run():
        code = run_setup()
        if code != 0: return code
    if args.command in ("train","benchmark","chat"):
        args = _merge(args.command, args)
    from .commands import cmd_doctor, cmd_benchmark, cmd_train, cmd_chat, cmd_data_validate, cmd_data_inspect
    if args.command == "doctor": return cmd_doctor(args)
    if args.command == "benchmark": return cmd_benchmark(args)
    if args.command == "train": return cmd_train(args)
    if args.command == "chat": return cmd_chat(args)
    if args.command == "data":
        return cmd_data_validate(args) if args.data_command == "validate" else cmd_data_inspect(args)
    raise SystemExit(f"Unknown command: {args.command}")
