import argparse

from core.compressor import (
    ImageCompressor
)

from config.settings import (
    BATCH_CLUSTERS
)


def print_report(r):

    print("\nResults")
    print("-"*40)

    for k,v in r["metrics"].items():

        print(
            f"{k}: "
            f"{round(v,4)}"
        )

    print(
      f"Runtime: "
      f"{round(r['runtime'],2)} sec"
    )

    print(
      f"Saved To: {r['output']}"
    )


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=32
    )

    parser.add_argument(
        "--output",
        default="output/compressed/result.jpg"
    )

    parser.add_argument(
        "--batch",
        action="store_true"
    )

    args=parser.parse_args()

    compressor=ImageCompressor(
        args.clusters
    )


    if args.batch:

        results=compressor.batch_compress(
            args.input,
            BATCH_CLUSTERS
        )

        for c,r in results:
            print(
                f"\n--- {c} Colors ---"
            )
            print_report(r)

    else:

        result=compressor.compress(
            args.input,
            args.output
        )

        print_report(result)


if __name__=="__main__":
    main()
