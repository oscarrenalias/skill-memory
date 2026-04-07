#!/usr/bin/env python3
import sys
import os
import subprocess
from pathlib import Path

_SKILL_DIR = Path(__file__).parent
_VENV = _SKILL_DIR / ".venv"
_DEPS = [
    "onnxruntime>=1.17.0",
    "sqlite-vec>=0.1.6",
    "tokenizers>=0.19.0",
    "numpy>=1.24.0",
]


def _bootstrap() -> None:
    if str(_VENV) in sys.prefix:
        return  # already running inside the managed venv
    venv_py = _VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not venv_py.exists():
        print("agent-memory: first run, installing dependencies…", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "venv", str(_VENV)])
        subprocess.check_call([str(venv_py), "-m", "pip", "install", "--quiet", *_DEPS])
    os.execv(str(venv_py), [str(venv_py)] + sys.argv)


_bootstrap()

import argparse  # noqa: E402 — must come after bootstrap


def cmd_init(args: argparse.Namespace) -> None:
    raise NotImplementedError("init not yet implemented")


def cmd_add(args: argparse.Namespace) -> None:
    raise NotImplementedError("add not yet implemented")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="memory.py",
        description="agent-memory: semantic long-term memory for Claude agents",
    )
    parser.add_argument("--db", metavar="PATH", help="Override DB path")
    sub = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")
    sub.required = True

    # init subcommand
    p_init = sub.add_parser("init", help="Initialise the memory database")
    p_init.add_argument("--db", metavar="PATH", help="Override DB path")
    p_init.set_defaults(func=cmd_init)

    # add subcommand
    p_add = sub.add_parser("add", help="Add a memory entry")
    p_add.add_argument("text", help="Text content to store")
    p_add.add_argument("--source", metavar="TAG", help="Optional origin tag or file path")
    p_add.add_argument("--meta", metavar="KEY=VALUE", action="append", default=[], help="Arbitrary metadata (repeatable)")
    p_add.add_argument("--db", metavar="PATH", help="Override DB path")
    p_add.set_defaults(func=cmd_add)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
