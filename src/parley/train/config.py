"""The training recipe. The settings experiments vary are Pydantic models, and the recipe values
no experiment varies are constants; to vary one, move it into a model and add a recipe version.
Performance choices with a concluded experiment are constants in parley.train.stack."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, assert_never

import yaml
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, TypeAdapter, model_validator

from parley.rows.layout import LAYOUT_L, Layout, has_own_voice

RESOLVED_CONFIG = "config.resolved.yaml"

BASE_MODEL = "google/gemma-4-12B-it"
#: The base weights every checkpoint's adapters were trained on.
BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
LORA_RANK = 64
LORA_ALPHA = 128
CODEBOOKS = 8
FEEDBACK_INIT_STD = 0.01
HOLDOUT_ROWS = 48
PAD_TO_MULTIPLE = 64
PREFETCH_WORKERS = 6
PREFETCH_DEPTH = 8
TEXT_KIND_WEIGHTS = MappingProxyType({"sil": 0.1, "eps": 1.0, "text": 1.5, "end": 1.5, "stop": 1.5, "hold": 0.1})
DEPTH_KIND_WEIGHTS = MappingProxyType({"sil": 1.0, "eps": 1.0, "text": 1.5, "end": 1.0, "stop": 1.0, "hold": 1.0})
LOG_EVERY = 10


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelCfg(Strict):
    #: Staged warm start: a checkpoint directory (relative to the volume) whose adapter,
    #: depth, feedback and token-row weights initialise this run; optimiser state and clocks
    #: start fresh (unlike a resume, which continues this run from its own latest save).
    warm_start: str | None = None
    #: Dropout on each adapter's input. A resumed or warm-started adapter takes this value, not the
    #: one saved with it.
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)


class SharedDepthCfg(Strict):
    layers: int = 4
    hidden: int = 1024


class CsmDepthCfg(Strict):
    #: The merged recipe keeps its depth shape, so it must be CSM-1B's decoder's.
    layers: Literal[4] = 4
    hidden: Literal[1024] = 1024
    kind: Literal["csm"]
    init: Literal["random", "sesame-csm-1b"]
    #: The head also reads Gemma's embeddings of the last `text_window` chosen text tokens, this
    #: frame's first, through a projection that starts at zero. 0 reads none.
    text_window: int = Field(default=0, ge=0)


def _depth_kind(value: Any) -> str:
    kind = value.get("kind") if isinstance(value, dict) else getattr(value, "kind", None)
    return "shared" if kind is None else str(kind)


#: Recipes and checkpoints written before the CSM head carry no `kind`.
DepthCfg = Annotated[
    Annotated[SharedDepthCfg, Tag("shared")] | Annotated[CsmDepthCfg, Tag("csm")],
    Discriminator(_depth_kind),
]


class DataCfg(Strict):
    """Rows trained in layout L."""

    rows: str  # a live dataset in datasets.yaml; launch refuses any other name
    #: Conversation-disjoint validation rows (a mix job's holdout dataset); None = no eval loss.
    holdout: str | None = None
    micro_batch: int = Field(default=4, ge=1)
    max_batch_frames: int = Field(default=3000, ge=1)  # per rank
    max_batch_positions: int | None = Field(default=None, ge=1)  # rows x longest padded row

    @property
    def frame_layout(self) -> Layout:
        return LAYOUT_L


class SpeechDataCfg(DataCfg):
    """Rows with a silent user channel, trained in the speech layout."""

    layout: Literal["speech"]
    positions: Literal["gapped", "contiguous"]

    @property
    def frame_layout(self) -> Layout:
        match self.positions:
            case "gapped":
                return "speech-gapped"
            case "contiguous":
                return "speech-contiguous"
            case unreachable:
                assert_never(unreachable)


class TalkerDataCfg(DataCfg):
    """Rows trained in layout L's positions with each reply written ahead for a separate talker,
    and none of the model's own audio fed back."""

    layout: Literal["talker"]

    @property
    def frame_layout(self) -> Layout:
        return "talker"


class RoomToneCfg(Strict):
    """Drawn per row and epoch. A row with no speech takes its floor from `quiet_dbfs`, and
    `slope` sets the noise's power falling as 1/f^slope."""

    share: float = Field(gt=0.0, le=1.0)
    snr_db: tuple[float, float]
    quiet_dbfs: tuple[float, float]
    slope: tuple[float, float]

    @model_validator(mode="after")
    def ranges(self) -> RoomToneCfg:
        if any(low > high for low, high in (self.snr_db, self.quiet_dbfs, self.slope)):
            raise ValueError("each room_tone range runs from its lowest value to its highest")
        if self.quiet_dbfs[1] >= 0.0 or self.slope[0] < 0.0:
            raise ValueError("room_tone sits below full scale, with a slope of at least 0")
        return self


class FloorDataCfg(DataCfg):
    """The talker layout with the floor held frame by frame: EPS keeps the talker playing, SIL
    while it plays stops it, and the stop is owed on every frame the user talks over it."""

    layout: Literal["talker-floor"]
    room_tone: RoomToneCfg | None = None

    @property
    def frame_layout(self) -> Layout:
        return "talker-floor"


def _data_layout(value: Any) -> str:
    layout = value.get("layout") if isinstance(value, dict) else getattr(value, "layout", None)
    return "L" if layout is None else str(layout)


#: Recipes and checkpoints written before the speech layout carry no `layout`.
AnyDataCfg = Annotated[
    Annotated[DataCfg, Tag("L")]
    | Annotated[SpeechDataCfg, Tag("speech")]
    | Annotated[TalkerDataCfg, Tag("talker")]
    | Annotated[FloorDataCfg, Tag("talker-floor")],
    Discriminator(_data_layout),
]


class OptimCfg(Strict):
    lr: float = 2.0e-4  # sqrt-scaled for global 16
    lora_lr_scale: float = 0.05
    depth_lr_scale: float = 1.0
    beta2: float = 0.999
    weight_decay: float = 0.01
    warmup_steps: int = 1200
    cosine_horizon_examples: int | None = None  # None = no decay
    min_lr_factor: float = 0.1
    grad_clip: float | None = None
    #: A step whose pre-clip gradient norm is a spike (spikes.py) leaves the weights and Adam's
    #: moments as they were.
    skip_spikes: bool = False


class LossCfg(Strict):
    codebook_weights: tuple[float, ...] = (2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    #: Scale on the depth loss's gradient into Gemma's hidden states; the depth head still trains
    #: at full strength. 0 leaves Gemma's adapters to the text loss alone.
    depth_grad_into_gemma: float = Field(default=1.0, ge=0.0, le=1.0)


class RunCfg(Strict):
    seed: int = 42
    max_steps: int | None = None
    max_examples: int | None = None
    max_seconds: int | None = Field(default=None, ge=1)
    save_every_examples: int = 16000
    save_every_steps: int | None = Field(default=None, ge=1)
    eval_every_steps: int = Field(default=1000, ge=1)  # holdout losses + grounding probe


class TrainConfig(Strict):
    model: ModelCfg = ModelCfg()
    #: A layout with a voice of its own trains a depth head; the talker layouts have none.
    depth: DepthCfg | None = SharedDepthCfg()
    data: AnyDataCfg
    optim: OptimCfg = OptimCfg()
    loss: LossCfg = LossCfg()
    run: RunCfg
    probe: Any | None = None

    @model_validator(mode="after")
    def stop_condition(self) -> TrainConfig:
        if self.run.max_steps is None and self.run.max_examples is None:
            raise ValueError("run.max_steps or run.max_examples must be set")
        if len(self.loss.codebook_weights) != CODEBOOKS:
            raise ValueError(f"loss.codebook_weights must have {CODEBOOKS} entries")
        own_voice = has_own_voice(self.data.frame_layout)
        if own_voice != (self.depth is not None):
            raise ValueError("a layout with a voice of its own needs a depth head; a talker, none")
        if not own_voice and self.probe is not None:
            raise ValueError("the feedback probe scores a voice of the model's own; a talker has none")
        return self


def load_yaml_mapping(path: str | Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return raw


def load_config(path: str | Path) -> TrainConfig:
    return TrainConfig.model_validate(load_yaml_mapping(path))


def resolved_config_text(cfg: BaseModel) -> str:
    """Any job config with every default filled in: the text saved beside each checkpoint."""
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)


def load_layout(checkpoint: Path) -> Layout:
    """Older configs' data sections can hold fields this code no longer has, so only the fields
    it knows are read."""
    path = checkpoint / RESOLVED_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the checkpoint records no resolved config")
    data = load_yaml_mapping(path)["data"]
    known = {key: value for key, value in data.items() if key in SpeechDataCfg.model_fields}
    return TypeAdapter(AnyDataCfg).validate_python(known).frame_layout


def load_depth_config(checkpoint: Path) -> DepthCfg | None:
    """The depth head's shape as the checkpoint's saved config records it, or None for a run with
    none. Older configs also record the backbone and codebook count, which must equal the
    constants the loader builds."""
    path = checkpoint / RESOLVED_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the checkpoint records no resolved config")
    raw = load_yaml_mapping(path)
    if raw["depth"] is None:
        return None
    depth = dict(raw["depth"])
    backbone = raw["model"].get("base_model", BASE_MODEL)
    codebooks = depth.pop("codebooks", CODEBOOKS)
    if (backbone, codebooks) != (BASE_MODEL, CODEBOOKS):
        raise ValueError(
            f"{path} records {backbone} with {codebooks} codebooks; this code builds {BASE_MODEL} with {CODEBOOKS}"
        )
    return TypeAdapter(DepthCfg).validate_python(depth)
