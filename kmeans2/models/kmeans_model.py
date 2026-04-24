import numpy as np
from sklearn.cluster import MiniBatchKMeans


class KMeansColorCompressor:
    """
    Performs image compression using color quantization.
    """

    def __init__(
        self,
        n_clusters=32,
        max_iter=300,
        batch_size=2048,
        random_state=42
    ):
        self.n_clusters = n_clusters

        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            batch_size=batch_size,
            random_state=random_state,
            n_init=10
        )


    def fit_predict(self, image):
        """
        image shape:
        (H,W,3)
        """

        h,w,c = image.shape

        pixels = image.reshape(-1,3)

        labels = self.model.fit_predict(pixels)

        centers = np.clip(
            self.model.cluster_centers_,
            0,
            255
        ).astype(np.uint8)

        compressed_pixels = centers[labels]

        compressed = compressed_pixels.reshape(
            h,w,c
        )

        return compressed


    def palette(self):
        return self.model.cluster_centers_
