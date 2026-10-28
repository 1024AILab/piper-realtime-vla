"""PiPER adapter derived from MAZHIPENG001/ROBOT_DATASET's single-arm recipe."""
import dataclasses
from pathlib import Path

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config
from openpi.training import weight_loaders
from processer import libero_policy

BASE = Path(__file__).resolve().parents[2]
PROMPT = 'Pick up the yellow duck and place it in the pen holder.'


@dataclasses.dataclass(frozen=True)
class PiperDuckDataConfig(config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        repack = transforms.Group(inputs=[transforms.RepackTransform({
            'image': 'image', 'wrist_image': 'wrist_image',
            'state': 'state', 'actions': 'actions', 'prompt': 'prompt',
        })])
        delta_mask = transforms.make_bool_mask(6, -1)
        data = transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        ).push(inputs=[transforms.DeltaActions(delta_mask)],
               outputs=[transforms.AbsoluteActions(delta_mask)])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=config.ModelTransformFactory()(model_config),
        )


def make_config():
    model = pi0_config.Pi0Config(
        pi05=True, paligemma_variant='gemma_2b_lora',
        action_expert_variant='gemma_300m_lora', action_horizon=50,
    )
    return config.TrainConfig(
        name='piper_duck_lora', project_name='piper-duck', exp_name='duck_en_v1',
        model=model, freeze_filter=model.get_freeze_filter(), ema_decay=None,
        data=PiperDuckDataConfig(repo_id='your-account/your-dataset',
            base_config=config.DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE/'base/pi05_base/params')),
        assets_base_dir=str(BASE/'assets'), checkpoint_base_dir=str(BASE/'checkpoints'),
        batch_size=4, num_workers=4, num_train_steps=30_000,
        log_interval=50, save_interval=1_000, keep_period=None,
        wandb_enabled=False, seed=42,
        policy_metadata={
            'robot_type':'piper', 'fps':10, 'prompt':PROMPT,
            'action_units':['0.001_degree']*6+['micrometer'],
            'state_semantics':'previous commanded actions (first frame: current command)',
            'camera_keys':['image','wrist_image'],
        },
    )
