from pathlib import Path

DOCS_PATH = Path("docs/traces")


def get_run_number(run_dir):
    try:
        return int(run_dir.name.split("-")[1])
    except (IndexError, ValueError):
        return -1


def main():
    runs = sorted(
        [d for d in DOCS_PATH.iterdir() if d.is_dir() and d.name.startswith("run-")], key=get_run_number, reverse=True
    )[:5]

    print("# Efficiency Traces\n")
    print("Below are the `trace_efficiency_mm.png` plots from the last 5 successful runs.\n")

    for run in runs:
        print(f"## {run.name}\n")
        for subdir in sorted(run.iterdir()):
            if subdir.is_dir():
                img_path = f"traces/{run.name}/{subdir.name}/trace_efficiency_mm.png"
                img_file = DOCS_PATH.parent / img_path
                if img_file.exists():
                    print(f"### {subdir.name}")
                    print(f"![{subdir.name}]({img_path})\n")


if __name__ == "__main__":
    main()
