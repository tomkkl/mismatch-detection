"""`agentlint` command line."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentlint")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("train", help="fine-tune the classifier")
    train.add_argument("config")
    scan = sub.add_parser("scan", help="check tools in a file or directory")
    scan.add_argument("path")
    data = sub.add_parser("data", help="build the generated dataset")
    data.add_argument("action", choices=["build"])
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
            print(json.dumps(run(cfg, args.tag), indent=2))
        except UsageLimitError as e:
            print(f"Usage limit reached: {e}\nRerun the same command later; cached calls "
                  "are reused.", file=sys.stderr)
            sys.exit(2)
    elif args.cmd == "train":
        from agentlint.model.train import train as run_train
        run_train(args.config)
    elif args.cmd == "scan":
        raise NotImplementedError


if __name__ == "__main__":
    main()
