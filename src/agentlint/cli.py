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
    data.add_argument("action", choices=["build", "collect", "second-opinion"])
    data.add_argument("config")
    data.add_argument("--tag", default="full", help="output subfolder, e.g. pilot")
    data.add_argument("--n-honest", type=int, help="override n_honest from the config")
    cov = sub.add_parser("coverage", help="Agent Audit coverage set: bugs, audit, report")
    cov.add_argument("action", choices=["bugs", "audit", "report"])
    cov.add_argument("config", help="e.g. configs/coverage.yaml")
    args = parser.parse_args()

    if args.cmd == "coverage":
        import json
        import shutil
        from pathlib import Path

        from agentlint.data.build import load_config
        cfg = load_config(args.config)
        if "domains_from" in cfg:
            cfg["domains"] = load_config(cfg["domains_from"])["domains"]
        if args.action == "bugs":
            from agentlint.data.bugs import run as run_bugs
            from agentlint.data.llm import ClaudeCLI
            gen = cfg["generator"]
            llm = ClaudeCLI(model=gen["model"], effort=gen["effort"], cache_dir=cfg["cache_dir"])
            print(json.dumps(run_bugs(cfg, llm), indent=2))
        elif args.action == "audit":
            from agentlint.eval.agent_audit import run as run_audit
            from agentlint.schema import ToolPair
            a = cfg["audit"]
            pairs = [ToolPair.from_dict(json.loads(line)) for path in a["pairs"]
                     for line in open(path, encoding="utf-8") if line.strip()]
            exe = shutil.which("agent-audit") or str(Path(sys_prefix()) / "Scripts" / "agent-audit")
            print(json.dumps(run_audit(pairs, Path(a["root"]), Path(a["out"]), exe), indent=2))
        else:
            from agentlint.eval.coverage import markdown, run as run_report
            print(markdown(run_report(cfg)))
        return

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
            if args.action == "second-opinion":
                from agentlint.data.collect import second_opinion
                from agentlint.data.llm import ClaudeCLI
                c = cfg["collect"]
                llm = ClaudeCLI(model=c["second_opinion_model"], effort=cfg["generator"]["effort"],
                                cache_dir=cfg["cache_dir"])
                print(json.dumps(second_opinion(c["out"], llm, cfg["generator"]["effort"]),
                                 indent=2))
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


def sys_prefix() -> str:
    import sys
    return sys.prefix


if __name__ == "__main__":
    main()
