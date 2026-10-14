import argparse
import os
import subprocess

from app.nwp.file_selector import select_latest_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=os.getenv("NWP_SSH_HOST", "nwp-host"))
    parser.add_argument("--directory", default=os.getenv("NWP_REMOTE_DIR", "/path/to/nwp/48h"))
    args = parser.parse_args()
    command = ["ssh", args.remote, f"find {args.directory} -maxdepth 1 -type f"]
    output = subprocess.check_output(command, text=True)
    selection = select_latest_files(output.splitlines())
    print("sfc:", selection.sfc, selection.issue_sfc_utc.isoformat())
    print("pl:", selection.pl, selection.issue_pl_utc.isoformat())


if __name__ == "__main__":
    main()
