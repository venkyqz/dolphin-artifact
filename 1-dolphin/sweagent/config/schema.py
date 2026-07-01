from urllib.parse import urlparse

from pydantic import BaseModel

from sweagent.run.batch_instances import (
    BatchInstance,
    SWEBenchInstances,
    BatchInstanceSourceConfig,
)
from sweagent.utils.telemetry import get_format_logger

logger = get_format_logger("schema")

dataset_mapping = {
    "full": "SWE-bench/SWE-Bench",
    "verified": "SWE-bench/SWE-Bench_Verified",
    "lite": "SWE-bench/SWE-bench_Lite",
    "multimodal": "SWE-bench/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
}


class SwebenchConfig(BaseModel):
    subset: str
    split: str
    problem_id: str

    def dataset(self):
        return dataset_mapping[self.subset]

    def uri(self):
        return f"swebench://{self.subset}.{self.split}/{self.problem_id}"

    def benchmark(self):
        bench = f"{self.subset}.{self.split}"
        return bench

    def load_subset_split(self) -> list[BatchInstance]:
        """
        Load all instances from the given config
        """
        # Initialize instance source config
        instance_source = SWEBenchInstances(
            subset=self.subset,
            split=self.split,
        )

        # Get instance configs
        instances = instance_source.get_instance_configs()
        logger.info(f"Instances: {instances}")
        return instances

    def load_swebench_instances(self) -> list[BatchInstance]:
        # Update issue text with problem ID
        logger.info("Processing swebench problem ID: {}", self.problem_id)

        # Initialize instance source config
        instance_source = SWEBenchInstances(
            subset=self.subset,
            split=self.split,
            instance_id=self.problem_id,
        )

        # Get instance configs
        instances = instance_source.get_instance_configs()
        logger.info(f"Instances: {instances}")
        return instances

    def load_swebench_instance_config(self) -> BatchInstanceSourceConfig:
        # Update issue text with problem ID
        logger.info("Processing swebench problem ID: {}", self.problem_id)

        # Initialize instance source config
        config = SWEBenchInstances(
            subset=self.subset,
            split=self.split,
            instance_id=self.problem_id,
        )

        return config


def parse_swebench_uri(uri: str) -> SwebenchConfig | None:
    """
    Parse a swebench URI and return the config if valid

    URI format:
    swebench://<subset>.<split>/<problem_id>

    known subset: lite, verify,full
    known split: train, dev, test
    """
    parsed_uri = urlparse(uri)

    if parsed_uri.scheme != "swebench":
        return None

    path_segments = parsed_uri.path.strip("/").split("/")
    problem_id = path_segments[-1] if path_segments else None

    subset, split = (
        parsed_uri.netloc.split(".") if "." in parsed_uri.netloc else (None, None)
    )

    if not all([subset, split]):
        return None

    return SwebenchConfig(subset=subset, split=split, problem_id=problem_id)
