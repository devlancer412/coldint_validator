# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Example miner / model training code.
# Do not expect much without extensive tuning of parameters or code updates!
# (Parameters depend heavily on the state of the model you're starting from)

import asyncio
import os
import wandb
import torch
import argparse
import constants
import dataset
from model import model_utils
import bittensor as bt
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer, Adafactor
import datetime as dt


from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TF32_OVERRIDE"] = "0"
os.environ["PYTORCH_NVFUSER_ENABLE_BF16"] = "1"

scaler = torch.amp.GradScaler("cuda")

# Set to bfloat16
torch_dtype = torch.bfloat16

# === Config ===
def get_config():
    """
    Set up and parse the command-line arguments to configure the system.

    Returns:
        A namespace object containing the configuration parameters.
    """

    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not launch a wandb run, does not send model to wandb, does not check if registered",
    )
    parser.add_argument(
        "--wandb_project", type=str, help="The wandb project to log to."
    )
    parser.add_argument("--wandb_entity", type=str, help="The wandb entity to log to.")
    parser.add_argument(
        "--model_dir",
        default=os.path.join(constants.ROOT_DIR, "local-models/"),
        help="Where to download/save models for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device on which to run. cpu or cuda",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=-1,
        help="Number of training epochs (-1 is infinite)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save model after this many epochs",
    )
    parser.add_argument("--pack-samples", default=False, action="store_true", help="Pack samples")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wdecay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument(
        "--bs", type=int, default=constants.batch_size, help="Batch size"
    )
    parser.add_argument(
        "--sl", type=int, default=2048, help="(Max) sequence length"
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=5,
        help="The number of training accumulation steps.",
    )
    parser.add_argument(
        "--pages_per_epoch",
        type=int,
        default=20,
        help="Number of pages trained on per epoch",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config

async def load_starting_model(
    path: str,
) -> LlamaForCausalLM:
    """Loads the model to train based on the provided config."""
    
    try:
        model = LlamaForCausalLM.from_pretrained(path)
        return model
    except:
        modelConfig = LlamaConfig.from_pretrained("luaqi/sn29_back_v16")

        model = LlamaForCausalLM(modelConfig)
        return model

async def load_starting_tokenizer(
    path: str
) -> AutoTokenizer:
    """Loads the model to train based on the provided config."""
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        return tokenizer
    except:
        tokenizer = AutoTokenizer.from_pretrained("coldint/phi3_stock_tokenizer")
        tokenizer.save_pretrained(path)
        return tokenizer

def freeze_layers(model, start_layer=1, end_layer=40):
    """
    Freezes the specified layers of the model by setting `requires_grad=False`.

    Args:
        model: The PyTorch model to modify.
        start_layer: The first layer index to freeze (inclusive).
        end_layer: The last layer index to freeze (inclusive).

    Returns:
        The model with the specified layers frozen.
    """
    # Ensure the model has a sequential or iterable structure for layers
    if hasattr(model, 'layers'):  # For models with a `layers` attribute
        layers = model.layers
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):  # For transformer-style encoders
        layers = model.encoder.layers
    else:
        raise ValueError("Model structure not recognized. Modify `freeze_layers` to access the correct layer hierarchy.")
    
    # Iterate over layers and freeze the specified range
    for i, layer in enumerate(layers):
        if start_layer <= i + 1 <= end_layer:  # PyTorch indices are 0-based, so adjust by +1
            for param in layer.parameters():
                param.requires_grad = False
    
    return model

async def main(config: bt.config):
    bt.logging(config=config)

    # Create bittensor objects if interaction with the chain is required
    # (no need to be registered)
    # wallet = subtensor = metagraph = remote_store = None
    # if config.load_uid or config.load_best:
    #     subtensor = bt.subtensor(config=config)
    #     remote_store = HuggingFaceModelStore()

    # Create a unique run id for this run.
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # model_dir = model_utils.model_path(config.model_dir, run_id)
    model_dir = f"/workspace/models/{config.model_dir}"
    tokenizer_dir = f"/workspace/tokenizers/{config.model_dir}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    use_wandb = False
    if not config.offline:
        if config.wandb_project is None or config.wandb_entity is None:
            bt.logging.warning(
                "Wandb project or entity not specified. This run will not be logged to wandb"
            )
        else:
            use_wandb = True

    # Init model.
    # metadata_store = ChainModelMetadataStore(subtensor, None, config.netuid)
    tokenizer: AutoTokenizer = await load_starting_tokenizer(tokenizer_dir)
    model: LlamaForCausalLM = await load_starting_model(model_dir)
    model.model = freeze_layers(model.model, start_layer=1, end_layer=40)
    if model is None or tokenizer is None:
        return False
    
    for param in model.parameters():
        param.requires_grad = True
    
    model = model.to(torch_dtype).train()
    model = model.to(config.device)
    # model = model.apply(lambda x: torch.utils.checkpoint.checkpoint(x))

    bt.logging.success(f"Saving model to path: {model_dir}.")
    model_utils.save(model, model_dir)
    # model_utils.save_tokenizer(tokenizer, tokenizer_dir)

    # Build optimizer
    # optimizer = Adafactor(
    #     model.parameters(), 
    #     # lr=config.lr, 
    #     warmup_init=True,
    #     scale_parameter=True, 
    #     relative_step=True
    # )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.wdecay)
    wandb_run = None

    # If using wandb, start a new run.
    if use_wandb:
        token = os.getenv("WANDB_API_KEY")
        if not token:
            raise ValueError(
                "To use Wandb, you must set WANDB_API_KEY in your .env file"
            )

        wandb.login(key=token)

        wandb_run = wandb.init(
            name=run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config={
                "run_name": run_id,
                "version": constants.__validator_version__,
                "type": "miner",
            },
            allow_val_change=True,
        )

        # At the end of the run, upload the model to wandb, for debugging purposes only.
        # This is not seen by validators.
        # wandb_run.save(os.path.join(model_dir, "*.*"), base_path=model_dir, policy="end")
    else:
        bt.logging.warning(
            "Not posting run to wandb. Either --offline is specified or the wandb settings are missing."
        )

    # Start the training loop
    epoch_step = 0
    global_step = 0
    n_acc_steps = 0
    accumulation_steps = config.accumulation_steps

    # scaler = torch.amp.GradScaler("cuda")
    model.gradient_checkpointing_enable()

    try:
        while epoch_step < config.num_epochs or config.num_epochs == -1:
            # Initialize loss accumulator for the epoch
            epoch_loss = 0.0

            # Prepare the data loader with random pages for each epoch
            bt.logging.success(
                f"Loading {config.pages_per_epoch} pages for training this epoch"
            )
            loader = dataset.SubsetFineWebEdu2Loader(
                batch_size=config.bs,
                sequence_length=config.sl,
                num_pages=config.pages_per_epoch,
                tokenizer=tokenizer,
                pack=config.pack_samples,
            )

            # Enumerate over the data loader
            n_batches = 0
            optimizer.zero_grad()  # Initialize gradients to zero
            
            for i, batch in enumerate(loader):
                # print(f"round: {i}")
                inputs = batch.to(model.device)

                # with torch.amp.autocast("cuda"): 
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss / accumulation_steps

                # scaler.scale(loss).backward()
                loss.backward()

                loss_detached = loss.detach().item()

                # del inputs, outputs, loss
                # torch.cuda.empty_cache()
            
                if (i + 1) % accumulation_steps == 0:
                    n_acc_steps += 1

                    # scaler.step(optimizer)
                    # scaler.update()
                    optimizer.step()  # Perform a single optimization step
                    optimizer.zero_grad()
                    
                    # print(f"Step: {n_acc_steps} loss: {loss_detached}")
                    if use_wandb:
                        wandb_run.log(
                            {"loss": loss_detached, "n_batches": n_batches},
                            step=n_acc_steps,
                        )

                n_batches += 1
                global_step += 1
                epoch_loss += loss_detached
                
                # del loss_detached
                torch.cuda.empty_cache()

            # Calculate the average loss for the epoch
            avg_loss = epoch_loss / n_batches
            
            wandb_run.log({ "avg_loss": avg_loss })

            # Log the average loss for the epoch
            bt.logging.success(f"Epoch: {epoch_step} average loss: {avg_loss}")
            epoch_step += 1

            if (epoch_step % config.save_interval) == 0:
                model_utils.save(model, model_dir)

                # del model
                # torch.cuda.empty_cache()
                # model = await load_starting_model(model_dir)
                # model.to(config.device)
                # model.gradient_checkpointing_enable()

        bt.logging.success(f"Finished training, saving model to {model_dir}")
        model_utils.save(model, model_dir)
        model_utils.save_tokenizer(tokenizer, tokenizer_dir)

    finally:
        # Important step.
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    # print('NOTICE: The base miner is left in this codebase for reference only. Remove this line if you want to actually run it.')
    # sys.exit(-1)
    # Parse and print configuration
    config = get_config()
    print(config)

    asyncio.run(main(config))
