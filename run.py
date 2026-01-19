#!/usr/bin/env python3
"""
VLA Training Framework CLI

Unified command-line interface for training, evaluation, and deployment.

Usage:
    python run.py train --mode finetune --preset pusht-finetune
    python run.py train --mode bc --env CartPole-v1
    python run.py train --mode ppo --env HumanoidWalk-v1
    python run.py eval --checkpoint ./checkpoints/model.pt
    python run.py infer --image robot.jpg --instruction "Pick up cube"
    python run.py export --checkpoint model.pt --format onnx
"""

import argparse
import sys
import os
from pathlib import Path


def train_command(args):
    """Train a VLA model with different training modes."""
    print("\n" + "=" * 60)
    print("VLA Training")
    print("=" * 60)

    # Dry run check
    if args.dry_run:
        print(f"[Dry run] Mode: {args.mode}")
        print(f"[Dry run] Would start training with the following configuration:")
        print(f"  - Preset: {args.preset}")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Learning Rate: {args.lr}")
        print(f"  - Batch Size: {args.batch_size}")
        return

    # Route to appropriate trainer based on mode
    mode = args.mode.lower()

    if mode == "finetune":
        _train_finetune(args)
    elif mode == "bc":
        _train_bc(args)
    elif mode == "ppo":
        _train_ppo(args)
    elif mode == "sac":
        _train_sac(args)
    elif mode == "iql":
        _train_iql(args)
    elif mode == "dagger":
        _train_dagger(args)
    elif mode == "gail":
        _train_gail(args)
    elif mode == "pretrain":
        _train_pretrain(args)
    elif mode == "demo":
        _train_demo(args)
    else:
        print(f"Unknown training mode: {mode}")
        print("Available modes: finetune, bc, dagger, gail, ppo, sac, iql, pretrain, demo")
        sys.exit(1)


def _train_finetune(args):
    """Fine-tune VLA model on robot manipulation data."""
    print(f"Mode: VLA Fine-tuning")
    print(f"Dataset: {args.dataset}")

    from train.finetune import VLAFineTuner, finetune_vla
    from config.training_config import FineTuningConfig
    from model.vla import VLAModel

    # Create config
    config = FineTuningConfig(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        freeze_vision=args.freeze_vision,
        freeze_llm=args.freeze_llm,
        use_lora=args.use_lora,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project or "vla-finetuning",
        experiment_name=args.experiment_name,
    )

    # Create model
    print(f"Creating VLA model...")
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        model = VLAModel.from_pretrained(args.checkpoint)
    else:
        model = VLAModel(
            vision_model_name=args.vision_model,
            llm_model_name=args.llm_model,
            action_dim=args.action_dim,
            freeze_vision=args.freeze_vision,
            freeze_llm=args.freeze_llm,
        )

    # Run fine-tuning
    finetune_vla(
        model=model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        freeze_vision=args.freeze_vision,
        freeze_llm=args.freeze_llm,
        use_lora=args.use_lora,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        experiment_name=args.experiment_name,
    )


def _train_bc(args):
    """Train with Behavioral Cloning."""
    print(f"Mode: Behavioral Cloning")
    print(f"Environment: {args.env}")

    from train.il import BehavioralCloning, VLABehavioralCloning
    from config.training_config import ILConfig
    import gymnasium as gym

    config = ILConfig(
        bc_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    if args.vla:
        # VLA-based BC
        from model.vla import VLAModel
        from train.datasets import create_lerobot_dataloader

        model = VLAModel(
            vision_model_name=args.vision_model,
            llm_model_name=args.llm_model,
            action_dim=args.action_dim,
        )

        train_loader = create_lerobot_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            split="train",
        )

        trainer = VLABehavioralCloning(
            model=model,
            config=config,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project or "vla-bc-training",
            experiment_name=args.experiment_name,
            dataset_name=args.dataset,
        )
        trainer.train(train_loader)
    else:
        # Standard BC with Gym environment
        env = gym.make(args.env)
        trainer = BehavioralCloning(
            env=env,
            config=config,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project or "bc-training",
            experiment_name=args.experiment_name,
        )
        # Train with expert policy (you can replace with your own)
        trainer.train(expert_policy=None, num_expert_episodes=args.num_expert_episodes)


def _train_ppo(args):
    """Train with PPO (Online RL)."""
    print(f"Mode: PPO (Online RL)")
    print(f"Environment: {args.env}")

    from train.online_rl import PPOTrainer
    from config.training_config import OnlineRLConfig
    import gymnasium as gym

    config = OnlineRLConfig(
        algorithm="ppo",
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        n_envs=args.num_envs,
    )

    env = gym.make(args.env)

    trainer = PPOTrainer(
        env=env,
        config=config,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project or "ppo-training",
        experiment_name=args.experiment_name,
    )
    trainer.train()


def _train_sac(args):
    """Train with SAC (Online RL)."""
    print(f"Mode: SAC (Online RL)")
    print(f"Environment: {args.env}")

    from train.online_rl import SACTrainer
    from config.training_config import OnlineRLConfig
    import gymnasium as gym

    config = OnlineRLConfig(
        algorithm="sac",
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    env = gym.make(args.env)

    trainer = SACTrainer(
        env=env,
        config=config,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project or "sac-training",
        experiment_name=args.experiment_name,
    )
    trainer.train()


def _train_iql(args):
    """Train with IQL (Offline RL)."""
    print(f"Mode: IQL (Offline RL)")
    print(f"Dataset: {args.dataset}")

    from train.offline_rl import IQLTrainer
    from config.training_config import OfflineRLConfig

    config = OfflineRLConfig(
        algorithm="iql",
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    trainer = IQLTrainer(
        dataset_name=args.dataset,
        config=config,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project or "iql-training",
        experiment_name=args.experiment_name,
    )
    trainer.train()


def _train_dagger(args):
    """Train with DAgger (Interactive Imitation Learning)."""
    print(f"Mode: DAgger (Interactive IL)")
    print(f"Environment: {args.env}")

    from train.il import DAgger, VLADAgger
    from config.training_config import ILConfig
    import gymnasium as gym

    config = ILConfig(
        dagger_iterations=args.dagger_iterations,
        dagger_episodes_per_iter=args.dagger_episodes_per_iter,
        dagger_beta_schedule=args.dagger_beta_schedule,
        dagger_initial_beta=args.dagger_initial_beta,
        bc_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    if args.vla:
        # VLA-based DAgger
        from model.vla import VLAModel
        from train.datasets import create_lerobot_dataloader

        model = VLAModel(
            vision_model_name=args.vision_model,
            llm_model_name=args.llm_model,
            action_dim=args.action_dim,
        )

        train_loader = create_lerobot_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            split="train",
        )

        # Expert function (returns ground truth actions from batch)
        expert_fn = lambda batch: batch.get("action", batch.get("actions"))

        trainer = VLADAgger(
            model=model,
            expert_fn=expert_fn,
            config=config,
        )
        trainer.train(train_loader, num_epochs_per_iter=args.epochs)
    else:
        # Standard DAgger with Gym environment
        env = gym.make(args.env)

        # Simple expert policy (replace with your own)
        from train.il.dagger import simple_expert_policy
        expert_policy = simple_expert_policy(args.env)

        trainer = DAgger(
            env=env,
            expert_policy=expert_policy,
            config=config,
        )
        trainer.train(num_initial_episodes=args.num_expert_episodes)


def _train_gail(args):
    """Train with GAIL (Adversarial Imitation Learning)."""
    print(f"Mode: GAIL (Adversarial IL)")
    print(f"Environment: {args.env}")

    from train.il import GAIL, VLAGAIL
    from config.training_config import ILConfig
    import gymnasium as gym

    config = ILConfig(
        gail_disc_hidden_dim=args.gail_disc_hidden_dim,
        gail_disc_updates=args.gail_disc_updates,
        gail_disc_lr=args.gail_disc_lr,
        gail_reward_scale=args.gail_reward_scale,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    if args.vla:
        # VLA-based GAIL
        from model.vla import VLAModel
        from train.datasets import create_lerobot_dataloader

        model = VLAModel(
            vision_model_name=args.vision_model,
            llm_model_name=args.llm_model,
            action_dim=args.action_dim,
        )

        train_loader = create_lerobot_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            split="train",
        )

        trainer = VLAGAIL(model=model, config=config)
        trainer.train(train_loader, total_steps=args.total_timesteps)
    else:
        # Standard GAIL with Gym environment
        env = gym.make(args.env)

        from train.il.gail import create_simple_expert
        expert_policy = create_simple_expert(args.env)

        trainer = GAIL(env=env, config=config)
        trainer.train(
            expert_policy=expert_policy,
            num_expert_episodes=args.num_expert_episodes,
            total_timesteps=args.total_timesteps,
        )


def _train_pretrain(args):
    """Pre-train VLM."""
    print(f"Mode: VLM Pre-training")

    from train.pretrain import VLMPretrainer
    from config.training_config import PretrainConfig

    config = PretrainConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    trainer = VLMPretrainer(
        vision_model_name=args.vision_model,
        llm_model_name=args.llm_model,
        config=config,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project or "vlm-pretraining",
        experiment_name=args.experiment_name,
    )
    trainer.train()


def _train_demo(args):
    """Run demo training (original behavior)."""
    print(f"Mode: Demo Training")

    from examples.pusht_demo import train as pusht_train, create_model, MockPushTDataset

    model, device = create_model(args.device)
    dataset = MockPushTDataset(num_samples=args.samples)
    pusht_train(model, dataset, device, num_epochs=args.epochs)


def eval_command(args):
    """Evaluate a trained model."""
    print("\n" + "=" * 60)
    print("VLA Evaluation")
    print("=" * 60)

    if not args.checkpoint:
        print("Error: --checkpoint required for evaluation")
        return

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num episodes: {args.episodes}")

    if args.mode == "env":
        # Evaluate in environment
        import gymnasium as gym
        import torch

        env = gym.make(args.env)

        # Load model based on checkpoint type
        if args.checkpoint.endswith(".pt"):
            from model.vla import VLAModel
            model = VLAModel.from_pretrained(args.checkpoint)
        else:
            # Assume it's a directory with model files
            from model.vla import VLAModel
            model = VLAModel.from_pretrained(args.checkpoint)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Run evaluation episodes
        rewards = []
        for ep in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            while not done:
                with torch.no_grad():
                    action = model.get_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
            print(f"Episode {ep + 1}: Reward = {total_reward:.2f}")

        import numpy as np
        print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    else:
        # Dataset-based evaluation (original demo)
        from examples.pusht_demo import evaluate, create_model, MockPushTDataset
        import torch

        model, device = create_model(args.device)
        dataset = MockPushTDataset(num_samples=100)

        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded checkpoint")

        evaluate(model, dataset, device)


def infer_command(args):
    """Run inference."""
    print("\n" + "=" * 60)
    print("VLA Inference")
    print("=" * 60)

    from infer import VLAInferenceEngine, InferenceConfig

    config = InferenceConfig(
        model_path=args.checkpoint,
        device=args.device,
        precision=args.precision,
    )

    engine = VLAInferenceEngine(config)

    if args.image:
        result = engine.predict(args.image, args.instruction)
        print(f"\nInstruction: {args.instruction}")
        print(f"Action: {result['action']}")
        print(f"Time: {result['inference_time_ms']:.2f} ms")

    if args.video:
        result = engine.process_video(args.video, args.instruction, args.output)
        print(f"\nProcessed {result['num_frames']} frames")
        print(f"FPS: {result['achievable_fps']:.1f}")

    if args.benchmark:
        engine.benchmark()


def export_command(args):
    """Export model to deployment format."""
    print("\n" + "=" * 60)
    print("Model Export")
    print("=" * 60)

    from model.utils import export_model, ExportConfig
    import torch
    import torch.nn as nn

    if not args.checkpoint:
        print("No checkpoint provided, creating demo model...")

        class DemoModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 7)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).flatten(1)
                return self.fc(x)

        model = DemoModel()
    else:
        # Load model from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # Would need to instantiate correct model architecture
        print("Loading model from checkpoint...")

    config = ExportConfig(
        output_dir=args.output_dir,
        model_name=args.name,
    )

    formats = args.format.split(",")
    exported = export_model(model, formats=formats, config=config)

    print(f"\nExported formats: {list(exported.keys())}")


def list_command(args):
    """List available presets and configs."""
    from config.hydra_config import list_presets

    print("\n" + "=" * 60)
    print("Available Presets")
    print("=" * 60)

    for preset in list_presets():
        print(f"  - {preset}")

    print("\n" + "=" * 60)
    print("Available Training Modes")
    print("=" * 60)
    print("  - finetune : VLA fine-tuning on robot manipulation data")
    print("  - bc       : Behavioral Cloning (imitation learning)")
    print("  - dagger   : DAgger (interactive imitation learning)")
    print("  - gail     : GAIL (adversarial imitation learning)")
    print("  - ppo      : PPO (online RL)")
    print("  - sac      : SAC (online RL)")
    print("  - iql      : IQL (offline RL)")
    print("  - pretrain : VLM pre-training")
    print("  - demo     : Simple demo training")

    print("\nUsage: python run.py train --mode <mode> [options]")


def demo_command(args):
    """Run demos."""
    print("\n" + "=" * 60)
    print("VLA Demos")
    print("=" * 60)

    if args.demo == "pusht":
        from examples.pusht_demo import run_demo
        run_demo()
    elif args.demo == "mujoco":
        from examples.mujoco_demo import main as mujoco_main
        sys.argv = ["mujoco_demo.py", "--env", args.env, "--train"]
        mujoco_main()
    elif args.demo == "carla":
        from examples.carla_demo import run_demo
        run_demo()
    elif args.demo == "inference":
        from examples.inference_demo import run_demo
        run_demo()
    else:
        print(f"Unknown demo: {args.demo}")
        print("Available: pusht, mujoco, carla, inference")


def main():
    parser = argparse.ArgumentParser(
        description="VLA Training Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a VLA model")

    # Training mode
    train_parser.add_argument("--mode", type=str, default="finetune",
                              choices=["finetune", "bc", "dagger", "gail", "ppo", "sac", "iql", "pretrain", "demo"],
                              help="Training mode (default: finetune)")

    # Config options
    train_parser.add_argument("--config", type=str, help="Path to config file")
    train_parser.add_argument("--preset", type=str, help="Use a preset config")

    # Model options
    train_parser.add_argument("--vision-model", type=str, default="google/siglip-base-patch16-224",
                              help="Vision encoder model")
    train_parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                              help="LLM model")
    train_parser.add_argument("--action-dim", type=int, default=7, help="Action dimension")
    train_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")

    # Training options
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")

    # Fine-tuning options
    train_parser.add_argument("--freeze-vision", action="store_true", help="Freeze vision encoder")
    train_parser.add_argument("--freeze-llm", action="store_true", help="Freeze LLM")
    train_parser.add_argument("--use-lora", action="store_true", help="Use LoRA")

    # Dataset/Environment options
    train_parser.add_argument("--dataset", type=str, default="lerobot/pusht", help="Dataset name")
    train_parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment")
    train_parser.add_argument("--vla", action="store_true", help="Use VLA model for BC")

    # RL-specific options
    train_parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total timesteps for RL")
    train_parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel envs")
    train_parser.add_argument("--num-expert-episodes", type=int, default=100, help="Expert episodes for BC/GAIL")

    # DAgger-specific options
    train_parser.add_argument("--dagger-iterations", type=int, default=10, help="DAgger iterations")
    train_parser.add_argument("--dagger-episodes-per-iter", type=int, default=20, help="Episodes per DAgger iteration")
    train_parser.add_argument("--dagger-beta-schedule", type=str, default="linear",
                              choices=["constant", "linear", "exponential"], help="Beta decay schedule")
    train_parser.add_argument("--dagger-initial-beta", type=float, default=1.0, help="Initial beta (expert mixing)")

    # GAIL-specific options
    train_parser.add_argument("--gail-disc-hidden-dim", type=int, default=256, help="GAIL discriminator hidden dim")
    train_parser.add_argument("--gail-disc-updates", type=int, default=5, help="Discriminator updates per policy update")
    train_parser.add_argument("--gail-disc-lr", type=float, default=3e-4, help="Discriminator learning rate")
    train_parser.add_argument("--gail-reward-scale", type=float, default=1.0, help="GAIL reward scale")

    # Logging options
    train_parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    train_parser.add_argument("--wandb-project", type=str, help="W&B project name")
    train_parser.add_argument("--experiment-name", type=str, help="Experiment name")

    # Misc options
    train_parser.add_argument("--device", type=str, default="auto", help="Device")
    train_parser.add_argument("--samples", type=int, default=1000, help="Training samples (demo mode)")
    train_parser.add_argument("--dry-run", action="store_true", help="Dry run")
    train_parser.set_defaults(func=train_command)

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Num episodes")
    eval_parser.add_argument("--device", type=str, default="auto", help="Device")
    eval_parser.add_argument("--mode", type=str, default="dataset", choices=["dataset", "env"],
                             help="Evaluation mode")
    eval_parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment for eval")
    eval_parser.set_defaults(func=eval_command)

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    infer_parser.add_argument("--image", type=str, help="Input image")
    infer_parser.add_argument("--video", type=str, help="Input video")
    infer_parser.add_argument("--instruction", type=str, default="Execute task",
                               help="Instruction")
    infer_parser.add_argument("--output", type=str, help="Output path")
    infer_parser.add_argument("--device", type=str, default="auto", help="Device")
    infer_parser.add_argument("--precision", type=str, default="fp32",
                               choices=["fp32", "fp16", "bf16"])
    infer_parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    infer_parser.set_defaults(func=infer_command)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    export_parser.add_argument("--format", type=str, default="torchscript",
                                help="Export format (onnx,torchscript,quantized)")
    export_parser.add_argument("--output-dir", type=str, default="./exported",
                                help="Output directory")
    export_parser.add_argument("--name", type=str, default="vla_model",
                                help="Model name")
    export_parser.set_defaults(func=export_command)

    # List command
    list_parser = subparsers.add_parser("list", help="List presets and modes")
    list_parser.set_defaults(func=list_command)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demos")
    demo_parser.add_argument("demo", type=str, nargs="?", default="pusht",
                              help="Demo name (pusht, mujoco, carla, inference)")
    demo_parser.add_argument("--env", type=str, default="CartPole-v1",
                              help="Environment for mujoco demo")
    demo_parser.set_defaults(func=demo_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
