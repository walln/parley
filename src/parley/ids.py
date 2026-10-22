from typing import Annotated

from pydantic import StringConstraints

DatasetName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*-v\d+\.\d+$")]
ExperimentId = Annotated[str, StringConstraints(pattern=r"^e\d{3,4}(-[a-z0-9]+)+$")]
GitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
VolumePath = Annotated[str, StringConstraints(pattern=r"^[\w-][\w.-]*(/[\w-][\w.-]*)*$")]
CHECKPOINT_PATTERN = r"^(e\d{3,4}(?:-[a-z0-9]+)+?)(?:-smoke/attempt-[1-9]\d*)?/step-\d+$"
CheckpointPath = Annotated[str, StringConstraints(pattern=CHECKPOINT_PATTERN)]
