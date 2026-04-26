import time
from pathlib import Path

from models.kmeans_model import (
    KMeansColorCompressor
)

from core.utils import (
    load_image,
    save_image,
    file_size
)

from core.metrics import report


class ImageCompressor:

    def __init__(self,
                 clusters=32):
        self.clusters=clusters


    def compress(
            self,
            input_path,
            output_path
    ):

        print("Loading image...")

        img=load_image(input_path)

        original_size=file_size(
            input_path
        )

        start=time.time()

        model=KMeansColorCompressor(
            n_clusters=self.clusters
        )

        compressed=model.fit_predict(
            img
        )

        save_image(
            output_path,
            compressed
        )

        elapsed=time.time()-start

        compressed_size=file_size(
            output_path
        )

        metrics=report(
            img,
            compressed,
            original_size,
            compressed_size
        )

        return {
            "runtime":elapsed,
            "metrics":metrics,
            "output":output_path
        }


    def batch_compress(
            self,
            input_path,
            cluster_list
    ):

        results=[]

        for c in cluster_list:

            out=Path(
                f"output/compressed/"
                f"kmeans_{c}.jpg"
            )

            self.clusters=c

            result=self.compress(
                input_path,
                out
            )

            results.append(
                (c,result)
            )

        return results
