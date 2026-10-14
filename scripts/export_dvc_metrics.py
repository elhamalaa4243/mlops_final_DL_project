import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

report = json.loads(Path(args.input).read_text(encoding="utf-8"))

metrics = {
    "test_accuracy": report["test_accuracy"],
    "test_macro_f1": report["test_macro_f1"],
    "test_ece": report["test_ece"],
}

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

print(json.dumps(metrics, indent=2))
