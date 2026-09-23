"""JEO config: Mamba-MTST segmentation on the ForTy v1 forest dataset.

Copy this file to ``jeo/configs/forest/mamba_forty_v1.py`` in a JEO checkout
and run:

    python -m jeo.train \
        --config jeo/configs/forest/mamba_forty_v1.py \
        --workdir /tmp/jeo/mamba_forty_v1

Config arguments (appended after a colon, e.g. ``...py:runlocal``):

    runlocal   a handful of steps with a tiny model, for smoke tests
    quick      a short run with a reduced model, for sanity checks
"""

from jeo.configs import config_utils
import ml_collections

NUM_CLASSES = 9
MODALITIES = ("optical", "sar_asc", "sar_desc", "climate", "elevation")


def _pp(train: bool) -> str:
  """Preprocessing pipeline string (see jeo_plugin/pp/forty_v1_ops.py)."""
  ops = ["forty_v1_normalize", "forty_v1_mask_invalid"]
  if train:
    ops.append("forty_v1_augment")
  ops += [
      "forty_v1_to_mamba(keep_metadata=False)",
      f"onehot({NUM_CLASSES}, key='labels', key_result='labels')",
      "keep('" + "','".join(MODALITIES) + "','labels')",
  ]
  return "|".join(ops)


def get_arg(arg):
  return config_utils.parse_arg(arg, runlocal=False, quick=False)


def get_config(arg=None):
  arg = get_arg(arg)
  config = ml_collections.ConfigDict()

  # --- Task -----------------------------------------------------------------
  config.task_type = "segmentation"
  config.task_kw = {"modalities": MODALITIES, "input_as_dict": True}
  config.num_classes = NUM_CLASSES

  # --- Data -----------------------------------------------------------------
  config.dataset = "forty_v1"
  config.dataset_dir = "gs://forest_typology"
  config.try_gcs = True
  config.train_split = "train"
  config.shuffle_buffer_size = 10_000
  config.pp_train = _pp(train=True)
  config.pp_eval = _pp(train=False)

  # --- Training -------------------------------------------------------------
  config.seed = 42
  config.batch_size = 32
  config.total_epochs = 100
  config.log_training_steps = 100
  config.log_eval_steps = 500
  config.ckpt_steps = 5000

  # --- Model ----------------------------------------------------------------
  config.model_name = "mamba_mtst"
  config.model = dict(
      mods=list(MODALITIES),
      patch_sizes={
          "optical": (2, 8, 8),
          "sar_asc": (2, 8, 8),
          "sar_desc": (2, 8, 8),
          "climate": (1, 1, 1),
          "elevation": (1, 8, 8),
      },
      d_model=384,
      d_state=32,
      temporal_depth=6,
      spatial_depth=6,
      dropout=0.1,
      num_classes=config.get_ref("num_classes"),
      head="segmentation",
  )

  # --- Optimisation ---------------------------------------------------------
  config.loss = "generalized_dice"
  config.optax_name = "big_vision.scale_by_adafactor"
  config.grad_clip_norm = 1.0
  config.wd = 1e-4
  config.lr = 5e-4
  config.schedule = dict(decay_type="cosine", warmup_steps=1000)

  # --- Evaluation -----------------------------------------------------------
  config.evals = ml_collections.ConfigDict()
  config.evals.val = ml_collections.ConfigDict(dict(
      type="semantic_segmentation",
      dataset=config.dataset,
      split="validation",
      pp=config.pp_eval,
      loss_name=config.loss,
      metrics=("mean_iou", "mean_dice", "pixel_acc"),
  ))

  # --- Short runs -----------------------------------------------------------
  if arg.runlocal:
    config.total_epochs = None  # total_steps and total_epochs are exclusive.
    config.total_steps = 5
    config.batch_size = 2
    config.log_training_steps = 1
    config.evals.val.steps = 2
    config.model["d_model"] = 16
    config.model["temporal_depth"] = 1
    config.model["spatial_depth"] = 1
    config.schedule["warmup_steps"] = 1

  if arg.quick:
    config.total_epochs = None
    config.total_steps = 100
    config.batch_size = 8
    config.log_training_steps = 10
    config.evals.val.steps = 20
    config.model["d_model"] = 128
    config.model["temporal_depth"] = 2
    config.model["spatial_depth"] = 2
    config.schedule["warmup_steps"] = 10

  return config


def metrics(*_):
  return ["val/mean_iou", "val/mean_dice", "training_loss"]
