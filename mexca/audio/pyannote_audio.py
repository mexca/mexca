# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# This module contains slightly modified code which is adapted from the
# pyannote.audio.pipelines.clustering and pyannote.audio.pipelines.speaker_diarization modules.
# It contains features that are currently only available in the development branch. Once these
# features are released, this module can be replaced.

"""Modified speaker diarization and clustering pipelines from the pyannote.audio package."""

import os
import random
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Text, Tuple, Union

import numpy as np
import yaml
from einops import rearrange
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from pyannote.audio import Audio, Inference, Model, Pipeline, __version__
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import CACHE_DIR
from pyannote.audio.core.pipeline import PIPELINE_PARAMS_NAME
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.speaker_verification import (
    PretrainedSpeakerEmbedding,
)
from pyannote.audio.pipelines.utils import PipelineModel, get_model
from pyannote.audio.utils.signal import binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.helper import get_class_by_name
from pyannote.database import FileFinder
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Categorical, Integer, ParamDict, Uniform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class BaseClustering(Pipeline):
    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        constrained_assignment: bool = False,
    ):
        super().__init__()
        self.metric = metric
        self.max_num_embeddings = max_num_embeddings
        self.constrained_assignment = constrained_assignment

    def set_num_clusters(
        self,
        num_embeddings: int,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
    ):
        min_clusters = num_clusters or min_clusters or 1
        min_clusters = max(1, min(num_embeddings, min_clusters))
        max_clusters = num_clusters or max_clusters or num_embeddings
        max_clusters = max(1, min(num_embeddings, max_clusters))

        if min_clusters > max_clusters:
            raise ValueError(
                f"min_clusters must be smaller than (or equal to) max_clusters "
                f"(here: min_clusters={min_clusters:g} and max_clusters={max_clusters:g})."
            )

        if min_clusters == max_clusters:
            num_clusters = min_clusters

        return num_clusters, min_clusters, max_clusters

    def filter_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter NaN embeddings and downsample embeddings

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.

        Returns
        -------
        filtered_embeddings : (num_embeddings, dimension) array
        chunk_idx : (num_embeddings, ) array
        speaker_idx : (num_embeddings, ) array
        """

        chunk_idx, speaker_idx = np.where(~np.any(np.isnan(embeddings), axis=2))

        # sample max_num_embeddings embeddings
        num_embeddings = len(chunk_idx)
        if num_embeddings > self.max_num_embeddings:
            indices = list(range(num_embeddings))
            random.shuffle(indices)
            indices = sorted(indices[: self.max_num_embeddings])
            chunk_idx = chunk_idx[indices]
            speaker_idx = speaker_idx[indices]

        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx

    def constrained_argmax(self, soft_clusters: np.ndarray) -> np.ndarray:
        soft_clusters = np.nan_to_num(
            soft_clusters, nan=np.nanmin(soft_clusters)
        )
        num_chunks, num_speakers, _ = soft_clusters.shape
        # num_chunks, num_speakers, num_clusters

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

        for c, cost in enumerate(soft_clusters):
            speakers, clusters = linear_sum_assignment(cost, maximize=True)
            for s, k in zip(speakers, clusters):
                hard_clusters[c, s] = k

        return hard_clusters

    def assign_embeddings(
        self,
        embeddings: np.ndarray,
        train_chunk_idx: np.ndarray,
        train_speaker_idx: np.ndarray,
        train_clusters: np.ndarray,
        constrained: bool = False,
    ):
        """Assign embeddings to the closest centroid

        Cluster centroids are computed as the average of the train embeddings
        previously assigned to them.

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension)-shaped array
            Complete set of embeddings.
        train_chunk_idx : (num_embeddings,)-shaped array
        train_speaker_idx : (num_embeddings,)-shaped array
            Indices of subset of embeddings used for "training".
        train_clusters : (num_embedding,)-shaped array
            Clusters of the above subset
        constrained : bool, optional
            Use constrained_argmax, instead of (default) argmax.

        Returns
        -------
        soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
        hard_clusters : (num_chunks, num_speakers)-shaped array
        centroids : (num_clusters, dimension)-shaped array
            Clusters centroids
        """

        # TODO: option to add a new (dummy) cluster in case num_clusters < max(frame_speaker_count)

        num_clusters = np.max(train_clusters) + 1
        num_chunks, num_speakers, _ = embeddings.shape

        train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]

        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        # compute distance between embeddings and clusters
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        soft_clusters = 2 - e2k_distance

        # assign each embedding to the cluster with the most similar centroid
        if constrained:
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        # NOTE: train_embeddings might be reassigned to a different cluster
        # in the process. based on experiments, this seems to lead to better
        # results than sticking to the original assignment.

        return hard_clusters, soft_clusters, centroids

    # pylint: disable=too-many-locals
    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature = None,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Has no effect when `num_clusters` is provided.

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        centroids : (num_clusters, dimension) array
            Centroid vectors of each cluster
        """

        (
            train_embeddings,
            train_chunk_idx,
            train_speaker_idx,
        ) = self.filter_embeddings(
            embeddings,
        )

        num_embeddings, _ = train_embeddings.shape
        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        if max_clusters < 2:
            # do NOT apply clustering when min_clusters = max_clusters = 1
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)

            return hard_clusters, soft_clusters, centroids

        train_clusters = self.cluster(
            train_embeddings,
            min_clusters,
            max_clusters,
            num_clusters=num_clusters,
        )

        hard_clusters, soft_clusters, centroids = self.assign_embeddings(
            embeddings,
            train_chunk_idx,
            train_speaker_idx,
            train_clusters,
            constrained=self.constrained_assignment,
        )

        return hard_clusters, soft_clusters, centroids


class CustomAgglomerativeClustering(BaseClustering):
    """Agglomerative clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".

    Hyper-parameters
    ----------------
    method : {"average", "centroid", "complete", "median", "single", "ward"}
        Linkage method.
    threshold : float in range [0.0, 2.0]
        Clustering threshold.
    min_cluster_size : int in range [1, 20]
        Minimum cluster size
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = np.inf,
        constrained_assignment: bool = False,
    ):
        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            constrained_assignment=constrained_assignment,
        )

        self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings
        self.method = Categorical(
            [
                "average",
                "centroid",
                "complete",
                "median",
                "single",
                "ward",
                "weighted",
            ]
        )

        # minimum cluster size
        self.min_cluster_size = Integer(1, 20)

    # pylint: disable=too-many-locals
    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = None,
    ):
        """

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) array
            Embeddings
        min_clusters : int
            Minimum number of clusters
        max_clusters : int
            Maximum number of clusters
        num_clusters : int, optional
            Actual number of clusters. Default behavior is to estimate it based
            on values provided for `min_clusters`,  `max_clusters`, and `threshold`.

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """

        num_embeddings, _ = embeddings.shape

        # heuristic to reduce self.min_cluster_size when num_embeddings is very small
        # (0.1 value is kind of arbitrary, though)
        min_cluster_size = min(
            self.min_cluster_size, max(1, round(0.1 * num_embeddings))
        )

        # linkage function will complain when there is just one embedding to cluster
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.uint8)

        # centroid, median, and Ward method only support "euclidean" metric
        # therefore we unit-normalize embeddings to somehow make them "euclidean"
        if self.metric == "cosine" and self.method in [
            "centroid",
            "median",
            "ward",
        ]:
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric="euclidean"
            )

        # other methods work just fine with any metric
        else:
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric=self.metric
            )

        # apply the predefined threshold
        clusters = (
            fcluster(dendrogram, self.threshold, criterion="distance") - 1
        )

        # split clusters into two categories based on their number of items:
        # large clusters vs. small clusters
        cluster_unique, cluster_counts = np.unique(
            clusters,
            return_counts=True,
        )
        large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        num_large_clusters = len(large_clusters)

        # force num_clusters to min_clusters in case the actual number is too small
        if num_large_clusters < min_clusters:
            num_clusters = min_clusters

        # force num_clusters to max_clusters in case the actual number is too large
        elif num_large_clusters > max_clusters:
            num_clusters = max_clusters

        if num_clusters is not None:
            # switch stopping criterion from "inter-cluster distance" stopping to "iteration index"
            _dendrogram = np.copy(dendrogram)
            _dendrogram[:, 2] = np.arange(num_embeddings - 1)

            best_iteration = num_embeddings - 1
            best_num_large_clusters = 1

            # traverse the dendrogram by going further and further away
            # from the "optimal" threshold

            for iteration in np.argsort(
                np.abs(dendrogram[:, 2] - self.threshold)
            ):
                # only consider iterations that might have resulted
                # in changing the number of (large) clusters
                new_cluster_size = _dendrogram[iteration, 3]
                if new_cluster_size < min_cluster_size:
                    continue

                # estimate number of large clusters at considered iteration
                clusters = (
                    fcluster(_dendrogram, iteration, criterion="distance") - 1
                )
                cluster_unique, cluster_counts = np.unique(
                    clusters, return_counts=True
                )
                large_clusters = cluster_unique[
                    cluster_counts >= min_cluster_size
                ]
                num_large_clusters = len(large_clusters)

                # keep track of iteration that leads to the number of large clusters
                # as close as possible to the target number of clusters.
                if abs(num_large_clusters - num_clusters) < abs(
                    best_num_large_clusters - num_clusters
                ):
                    best_iteration = iteration
                    best_num_large_clusters = num_large_clusters

                # stop traversing the dendrogram as soon as we found a good candidate
                if num_large_clusters == num_clusters:
                    break

            # re-apply best iteration in case we did not find a perfect candidate
            if best_num_large_clusters != num_clusters:
                clusters = (
                    fcluster(_dendrogram, best_iteration, criterion="distance")
                    - 1
                )
                cluster_unique, cluster_counts = np.unique(
                    clusters, return_counts=True
                )
                large_clusters = cluster_unique[
                    cluster_counts >= min_cluster_size
                ]
                num_large_clusters = len(large_clusters)
                print(
                    f"Found only {num_large_clusters} clusters. Using a smaller value than {min_cluster_size} for `min_cluster_size` might help."
                )

        if num_large_clusters == 0:
            clusters[:] = 0
            return clusters

        small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        if len(small_clusters) == 0:
            return clusters

        # re-assign each small cluster to the most similar large cluster based on their respective centroids
        large_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == large_k], axis=0)
                for large_k in large_clusters
            ]
        )
        small_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == small_k], axis=0)
                for small_k in small_clusters
            ]
        )
        centroids_cdist = cdist(
            large_centroids, small_centroids, metric=self.metric
        )
        for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
            clusters[clusters == small_clusters[small_k]] = large_clusters[
                large_k
            ]

        # re-number clusters from 0 to num_large_clusters
        _, clusters = np.unique(clusters, return_inverse=True)
        return clusters


class CustomClustering(Enum):
    CustomAgglomerativeClustering = CustomAgglomerativeClustering


class CustomPipeline(SpeakerDiarization):
    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation@2022.07",
        segmentation_step: float = 0.1,
        embedding: PipelineModel = "speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e",
        embedding_exclude_overlap: bool = False,
        clustering: str = "CustomAgglomerativeClustering",
        embedding_batch_size: int = 1,
        segmentation_batch_size: int = 1,
        der_variant: dict = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super(SpeakerDiarization, self).__init__()

        self.segmentation_model = segmentation
        model: Model = get_model(segmentation, use_auth_token=use_auth_token)

        self.segmentation_step = segmentation_step

        self.embedding = embedding
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap

        self.klustering = clustering

        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}

        segmentation_duration = model.specifications.duration
        self._segmentation = Inference(
            model,
            duration=segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
            batch_size=segmentation_batch_size,
        )
        self._frames: SlidingWindow = (
            self._segmentation.model.introspection.frames
        )

        self.segmentation = ParamDict(
            threshold=Uniform(0.1, 0.9),
            min_duration_off=Uniform(0.0, 1.0),
        )

        if self.klustering == "OracleClustering":
            metric = "not_applicable"

        else:
            self._embedding = PretrainedSpeakerEmbedding(
                self.embedding, use_auth_token=use_auth_token
            )
            self._audio = Audio(
                sample_rate=self._embedding.sample_rate, mono="downmix"
            )
            metric = self._embedding.metric

        try:
            Klustering = CustomClustering[clustering]
        except KeyError as exc:
            raise ValueError(
                f'clustering must be one of [{", ".join(list(CustomClustering.__members__))}]'
            ) from exc
        self.clustering = Klustering.value(metric=metric)

    # pylint: disable=too-many-locals
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[Text, Path],
        hparams_file: Union[Text, Path] = None,
        use_auth_token: Union[Text, None] = None,
        cache_dir: Union[Path, Text] = CACHE_DIR,
    ) -> "Pipeline":
        """Load pretrained pipeline

        Parameters
        ----------
        checkpoint_path : Path or str
            Path to pipeline checkpoint, or a remote URL,
            or a pipeline identifier from the huggingface.co model hub.
        hparams_file: Path or str, optional
        use_auth_token : str, optional
            When loading a private huggingface.co pipeline, set `use_auth_token`
            to True or to a string containing your hugginface.co authentication
            token that can be obtained by running `huggingface-cli login`
        cache_dir: Path or str, optional
            Path to model cache directory. Defauorch/pyannote" when unset.
        """

        checkpoint_path = str(checkpoint_path)

        if os.path.isfile(checkpoint_path):
            config_yml = checkpoint_path

        else:
            if "@" in checkpoint_path:
                model_id = checkpoint_path.split("@", maxsplit=1)[0]
                revision = checkpoint_path.split("@")[1]
            else:
                model_id = checkpoint_path
                revision = None

            try:
                config_yml = hf_hub_download(
                    model_id,
                    PIPELINE_PARAMS_NAME,
                    repo_type="model",
                    revision=revision,
                    library_name="pyannote",
                    library_version=__version__,
                    cache_dir=cache_dir,
                    use_auth_token=use_auth_token,
                )

            except RepositoryNotFoundError:
                print(
                    f"""
Could not download '{model_id}' pipeline.
It might be because the pipeline is private or gated so make
sure to authenticate. Visit https://hf.co/settings/tokens to
create your access token and retry with:

   >>> Pipeline.from_pretrained('{model_id}',
   ...                          use_auth_token=YOUR_AUTH_TOKEN)

If this still does not work, it might be because the pipeline is gated:
visit https://hf.co/{model_id} to accept the user conditions."""
                )
                return None

        with open(config_yml, "r", encoding="utf-8") as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)

        # initialize pipeline
        Klass = cls
        params = config["pipeline"].get("params", {})
        params.setdefault("use_auth_token", use_auth_token)
        params["clustering"] = "CustomAgglomerativeClustering"
        pipeline = Klass(**params)

        # freeze  parameters
        if "freeze" in config:
            params = config["freeze"]
            pipeline.freeze(params)

        if "params" in config:
            pipeline.instantiate(config["params"])

        if hparams_file is not None:
            pipeline.load_params(hparams_file)

        if "preprocessors" in config:
            preprocessors = {}
            for key, preprocessor in config.get("preprocessors", {}).items():
                # preprocessors:
                #    key:
                #       name: package.module.ClassName
                #       params:
                #          param1: value1
                #          param2: value2
                if isinstance(preprocessor, dict):
                    Klass = get_class_by_name(
                        preprocessor["name"],
                        default_module_name="pyannote.audio",
                    )
                    params = preprocessor.get("params", {})
                    preprocessors[key] = Klass(**params)
                    continue

                try:
                    # preprocessors:
                    #    key: /path/to/database.yml
                    preprocessors[key] = FileFinder(database_yml=preprocessor)

                except FileNotFoundError:
                    # preprocessors:
                    #    key: /path/to/{uri}.wav
                    template = preprocessor
                    preprocessors[key] = template

            pipeline.preprocessors = preprocessors

        return pipeline

    # pylint: disable=arguments-renamed
    def apply(
        self,
        file: AudioFile,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
        return_embeddings: bool = False,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Number of speakers, when known.
        min_speakers : int, optional
            Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers : int, optional
            Maximum number of speakers. Has no effect when `num_speakers` is provided.
        return_embeddings : bool, optional
            Return representative speaker embeddings.
        hook : callable, optional
            Callback called after each major steps of the pipeline as follows:
                hook(step_name,      # human-readable name of current step
                     step_artefact,  # artifact generated by current step
                     file=file)      # file being processed
            Time-consuming steps call `hook` multiple times with the same `step_name`
            and additional `completed` and `total` keyword arguments usable to track
            progress of current step.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        embeddings : np.array, optional
            Representative speaker embeddings such that `embeddings[i]` is the
            speaker embedding for i-th speaker in diarization.labels().
            Only returned when `return_embeddings` is True.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        num_speakers, min_speakers, max_speakers = self.set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        segmentations = self.get_segmentations(file)
        hook("segmentation", segmentations)
        #   shape: (num_chunks, num_frames, local_num_speakers)

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            segmentations,
            onset=self.segmentation.threshold,
            frames=self._frames,
            warm_up=(0.0, 0.0),
        )
        hook("speaker_counting", count)
        #   shape: (num_frames, 1)
        #   dtype: int

        # exit early when no speaker is ever active
        if np.nanmax(count.data) == 0.0:
            diarization = Annotation(uri=file["uri"])
            if return_embeddings:
                return diarization, np.zeros((0, self._embedding.dimension))

            return diarization

        # binarize segmentation
        binarized_segmentations: SlidingWindowFeature = binarize(
            segmentations,
            onset=self.segmentation.threshold,
            initial_state=False,
        )

        if self.klustering == "OracleClustering" and not return_embeddings:
            embeddings = None
        else:
            embeddings = self.get_embeddings(
                file,
                binarized_segmentations,
                exclude_overlap=self.embedding_exclude_overlap,
            )
            hook("embeddings", embeddings)
            #   shape: (num_chunks, local_num_speakers, dimension)

        hard_clusters, _, centroids = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
            file=file,  # <== for oracle clustering
            frames=self._frames,  # <== for oracle clustering
        )
        # hard_clusters: (num_chunks, num_speakers)
        # centroids: (num_speakers, dimension)

        # reconstruct discrete diarization from raw hard clusters

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        hard_clusters[inactive_speakers] = -2
        discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        hook("discrete_diarization", discrete_diarization)

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=self.segmentation.min_duration_off,
        )
        diarization.uri = file["uri"]

        # at this point, `diarization` speaker labels are integers
        # from 0 to `num_speakers - 1`, aligned with `centroids` rows.

        if "annotation" in file and file["annotation"]:
            # when reference is available, use it to map hypothesized speakers
            # to reference speakers (this makes later error analysis easier
            # but does not modify the actual output of the diarization pipeline)
            _, mapping = self.optimal_mapping(file["annotation"], diarization)

            # in case there are more speakers in the hypothesis than in
            # the reference, those extra speakers are missing from `mapping`.
            # we add them back here
            mapping = {
                key: mapping.get(key, key) for key in diarization.labels()
            }

        else:
            # when reference is not available, rename hypothesized speakers
            # to human-readable SPEAKER_00, SPEAKER_01, ...
            mapping = dict(zip(diarization.labels(), self.classes()))

        diarization = diarization.rename_labels(mapping=mapping)

        # at this point, `diarization` speaker labels are strings (or mix of
        # strings and integers when reference is available and some hypothesis
        # speakers are not present in the reference)

        if not return_embeddings:
            return diarization

        # re-order centroids so that they match
        # the order given by diarization.labels()
        inverse_mapping = {label: index for index, label in mapping.items()}
        centroids = centroids[
            [inverse_mapping[label] for label in diarization.labels()]
        ]

        # FIXME: the number of centroids may be smaller than the number of speakers
        # in the annotation. This can happen if the number of active speakers
        # obtained from `speaker_count` for some frames is larger than the number
        # of clusters obtained from `clustering`. Will be fixed in the future

        return diarization, centroids

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(**self.der_variant)
