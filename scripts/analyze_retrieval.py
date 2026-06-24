import argparse
import json
from dataclasses import asdict
from pathlib import Path

from knowledge_base_assistant.application.retrieval_evaluation import (
    evaluate_bm25_queries,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze BM25 retrieval errors."
    )
    parser.add_argument(
        "golden_path",
        type=Path,
    )
    parser.add_argument(
        "chunks_path",
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/retrieval_evaluation.jsonl"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--only-misses",
        action="store_true",
        help="Write only queries without a relevant chunk in top-k.",
    )

    args = parser.parse_args()

    results = evaluate_bm25_queries(
        golden_path=args.golden_path,
        chunks_path=args.chunks_path,
        top_k=args.top_k,
    )

    selected_results = (
        tuple(
            result
            for result in results
            if result.hit_rate_at_k == 0.0
        )
        if args.only_misses
        else results
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for result in selected_results:
            output_file.write(
                json.dumps(
                    asdict(result),
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Evaluated queries: {len(results)}")
    print(f"Written queries: {len(selected_results)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()