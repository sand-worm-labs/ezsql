import argparse
from pathlib import Path
from typing import Iterator


def iter_yaml_files(path: Path) -> Iterator[Path]:
    """Recursively find all YAML files in the given path."""
    if path.is_file() and path.suffix in {".yml", ".yaml"}:
        yield path
        return

    if path.is_dir():
        for yaml_path in sorted(path.rglob("*.yml")):
            yield yaml_path
        for yaml_path in sorted(path.rglob("*.yaml")):
            yield yaml_path


def extract_tables(yaml_path: Path) -> list[dict[str, str]]:
    """
    Manually parse dbt source YAML files by indentation to extract 
    source names and their associated table names.
    """
    results: list[dict[str, str]] = []
    current_source = ""
    in_tables = False

    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        # Calculate indentation level
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        # Source name level (typically indent 2 in dbt sources)
        if indent == 2 and stripped.startswith("- name:"):
            current_source = stripped.split(":", 1)[1].strip()
            in_tables = False
            continue

        # Start of the tables block
        if indent == 4 and stripped == "tables:":
            in_tables = True
            continue

        # Individual table name level (typically indent 6)
        if in_tables and indent == 6 and stripped.startswith("- name:"):
            table_name = stripped.split(":", 1)[1].strip()
            results.append(
                {
                    "file": str(yaml_path),
                    "source": current_source,
                    "table": table_name,
                }
            )
            continue

        # If we hit a line with lower indentation, we've exited the tables block
        if in_tables and indent <= 4 and stripped != "tables:":
            in_tables = False

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract dbt source table names from source YAML files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="domains/spellbook/sources",
        help="YAML file or directory to scan. Defaults to domains/spellbook/sources",
    )
    parser.add_argument(
        "--format",
        choices=("table", "source_table", "file_source_table"),
        default="source_table",
        help="Output format. Defaults to source_table (namespace,table_name)",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        default=True,
        help="Only print unique output lines. (Default: True)",
    )
    parser.add_argument(
        "--no-unique",
        action="store_false",
        dest="unique",
        help="Print all output lines including duplicates.",
    )
    args = parser.parse_args()

    target = Path(args.path)
    files = list(iter_yaml_files(target))
    if not files:
        raise SystemExit(f"No YAML files found under: {target}")

    # Set Header based on selected format
    if args.format == "table":
        header = "table_name"
    elif args.format == "source_table":
        header = "namespace,table_name"
    else:
        header = "file_path,namespace,table_name"

    output_lines: list[str] = []
    for yaml_file in files:
        for item in extract_tables(yaml_file):
            if args.format == "table":
                output_lines.append(item["table"])
            elif args.format == "source_table":
                output_lines.append(f'{item["source"]},{item["table"]}')
            else:
                output_lines.append(
                    f'{item["file"]},{item["source"]},{item["table"]}'
                )

    if args.unique:
        output_lines = sorted(set(output_lines))

    # Print the CSV-ready output
    print(header)
    for line in output_lines:
        print(line)


if __name__ == "__main__":
    main()