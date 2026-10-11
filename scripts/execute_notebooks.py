"""Execute and verify all chapter notebooks from the repository root."""

from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = sorted(ROOT.glob("Chapter_*/notebook.ipynb"))
    if sys.argv[1:]:
        selected = {int(value) for value in sys.argv[1:]}
        paths = [path for path in paths if int(path.parent.name.split("_")[1]) in selected]
    if not paths:
        raise RuntimeError("No chapter notebooks found")
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        errors = [
            output
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]
        if errors:
            raise RuntimeError(f"{path}: {len(errors)} errors")
        for cell in notebook.cells:
            cell.metadata.pop("execution", None)  # omit run-specific timestamps
        nbformat.write(notebook, path)
        print(f"PASS {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
