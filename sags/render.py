import sys
from contextlib import ExitStack
from typing import Union, Tuple
import logging
from pathlib import Path
import os
import json
from functools import partial

import click
from .datasets import load_dataset
from nerfbaselines import Method, get_method_spec, build_method_class
from nerfbaselines import backends
from .evaluation import render_all_images, DefaultEvaluationProtocol
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from nerfbaselines.cli._common import handle_cli_error, click_backend_option, NerfBaselinesCliCommand
from .datasets.phototourism import load_dfc2019_dataset, NerfWEvaluationProtocol
from .method import SAGS
from .types import FrozenSet, Method, Dataset, DatasetFeature, EvaluationProtocol, Logger
from . import datasets

@click.command("render")
@click.option("--checkpoint", default=None, required=True, type=str, help=(
    "Path to the checkpoint directory. It can also be a remote path (starting with `http(s)://`) or be a path inside a zip file."
))
@click.option("--data", type=str, required=True, help=(
    "A path to the dataset to render the cameras from. The dataset can be either an external dataset (e.g., a path starting with `external://{dataset}/{scene}`) or a local path to a dataset. If the dataset is an external dataset, the dataset will be downloaded and cached locally. If the dataset is a local path, the dataset will be loaded directly from the specified path."))
@click.option("--output", type=str, default="predictions", help="Output directory or tar.gz/zip file.")
@click.option("--split", type=str, default="test", show_default=True, help="Dataset split to render.")
@click_backend_option()
@handle_cli_error
def render_command(checkpoint: str, data: str, output: str, split: str, backend_name):

    checkpoint = str(checkpoint)

    # if os.path.exists(output):
    #     logging.critical("Output path already exists")
    #     sys.exit(1)

    with ExitStack() as stack:
        # Open checkpoint directory
        _checkpoint_path = stack.enter_context(open_any_directory(checkpoint, mode="r"))
        stack.enter_context(backends.mount(_checkpoint_path, _checkpoint_path))
        checkpoint_path = Path(_checkpoint_path)

        # Read method nb-info
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"

        method = SAGS(
            checkpoint=checkpoint_path,
            train_dataset=None,
            config_overrides=None,
        )
        
        if 1:
            evaluation_protocol = NerfWEvaluationProtocol()
            embedding_index = None
        else:
            evaluation_protocol = DefaultEvaluationProtocol()
            embedding_index = {"JAX_068_002_RGB.tif": 0, "JAX_068_009_RGB.tif": 8, "JAX_068_013_RGB.tif": 0, "JAX_068_020_RGB.tif": 8}

        load_dataset_fn = partial(
            load_dataset,
            load_dataset_fn=load_dfc2019_dataset,
            download_dataset_fn=None,
            evaluation_protocol=evaluation_protocol.get_name(),
        )
        features: FrozenSet[DatasetFeature] = frozenset({"color", "points3D_xyz"})
        test_dataset = load_dataset_fn(data, "test", features, load_features=False)
        method_info = SAGS.get_method_info()
        supported_camera_models = method_info.get("supported_camera_models", frozenset(("pinhole", "affine")))
        test_dataset = datasets.dataset_load_features(test_dataset, supported_camera_models=supported_camera_models)
        test_dataset["images"] = [x[..., :3] for x in test_dataset["images"]]

        # print(test_dataset['images'])
        
        # Render all images

        for _ in render_all_images(method, test_dataset, output=output, evaluation_protocol=evaluation_protocol, embedding_index=embedding_index):
            pass

if __name__ == "__main__":
    render_command()  # pylint: disable=no-value-for-parameter
    