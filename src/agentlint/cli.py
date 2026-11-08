"""`agentlint` command line."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentlint")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train", help="fine-tune the classifier")
    train.add_argument("config")
    score = sub.add_parser("score", help="score a saved checkpoint on a test set (once)")
    score.add_argument("checkpoint", help="e.g. checkpoints/binary")
    score.add_argument("pairs", help="e.g. data/splits/full/test.jsonl")
    score.add_argument("--tag", help="results subfolder (default: <dir>-<file>, e.g. full-test)")
    score.add_argument("--force", action="store_true", help="overwrite an existing result")
    score.add_argument("--batch-size", type=int, help="override batch_size from the checkpoint config")
    scan = sub.add_parser("scan", help="check tools in a file or directory")
    scan.add_argument("path")
    data = sub.add_parser("data", help="build the generated dataset, or collect real tools")
    data.add_argument("action", choices=["build", "collect"])
    data.add_argument("config")
    data.add_argument("--tag", default="full", help="output subfolder, e.g. pilot")
    data.add_argument("--n-honest", type=int, help="override n_honest from the config")
    args = parser.parse_args()

    if args.cmd == "data":
        import json
        import sys

        from agentlint.data.build import load_config, run
        from agentlint.data.llm import UsageLimitError
        cfg = load_config(args.config)
        if args.n_honest:
            cfg["n_honest"] = args.n_honest
        try:
            if args.action == "collect":
                from agentlint.data.collect import collect
                from agentlint.data.llm import ClaudeCLI
                gen = cfg["generator"]
                llm = ClaudeCLI(model=gen["model"], effort=gen["effort"],
                                cache_dir=cfg["cache_dir"])
                print(json.dumps(collect(cfg, llm), indent=2))
                return
            print(json.dumps(run(cfg, args.tag), indent=2))
        except UsageLimitError as e:
            print(f"Usage limit reached: {e}\nRerun the same command later; cached calls "
                  "are reused.", file=sys.stderr)
            sys.exit(2)
    elif args.cmd == "train":
        from agentlint.model.train import train as run_train
        run_train(args.config)
    elif args.cmd == "score":
        import sys

        from agentlint.eval.score import score as run_score
        try:
            run_score(args.checkpoint, args.pairs, args.tag, args.force, args.batch_size)
        except FileExistsError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
    elif args.cmd == "scan":
        raise NotImplementedError


if __name__ == "__main__":
    main()
