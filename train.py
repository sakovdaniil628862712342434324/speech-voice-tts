import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from data_loader import CustomDataSet, collate_fn
from tqdm import tqdm
import wandb
import argparse
import os

def train(args):
    # --- Setup ---
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU.")
        args.device = 'cpu'

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Tokenizer ---
    print("Loading tokenizer...")
    # Load tokenizer from the directory specified, assuming config files are there
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
         # Ensure PAD token is set
        if tokenizer.pad_token is None:
             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             print("Added [PAD] token to tokenizer.")
    except Exception as e:
        print(f"Error loading tokenizer from {args.tokenizer_path}: {e}")
        print("Ensure tokenizer files (tokenizer.json, config.json, etc.) are present.")
        return

    pad_token_id = tokenizer.pad_token_id

    # --- Load Model ---
    print("Loading model...")
    if args.checkpoint_path:
        print(f"Resuming training from checkpoint: {args.checkpoint_path}")
        try:
            # Load model from the checkpoint directory
            model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path)
            print("Model loaded from checkpoint.")
        except Exception as e:
            print(f"Error loading model from checkpoint {args.checkpoint_path}: {e}")
            return
    else:
        print("Initializing new model from config...")
        try:
            config = GPT2Config.from_pretrained(args.config_path)
            # Ensure vocab size matches tokenizer after potential additions
            if config.vocab_size != len(tokenizer):
                 print(f"Config vocab size ({config.vocab_size}) doesn't match tokenizer ({len(tokenizer)}). Adjusting config.")
                 config.vocab_size = len(tokenizer)
            model = GPT2LMHeadModel(config)
            # Resize embeddings if we added tokens AFTER initial config creation
            model.resize_token_embeddings(len(tokenizer))
            print("New model initialized.")
        except Exception as e:
            print(f"Error initializing model from config {args.config_path}: {e}")
            return

    model.to(args.device)

    # --- Load Data ---
    print("Setting up dataset...")
    # Assuming 'picked_data' dir contains the preprocessed HF dataset
    try:
        train_ds = CustomDataSet(
            dataset_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            snac_model_path=args.snac_model_path,
            max_length=model.config.n_positions, # Use model's max length
            device=args.device
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn # Use custom collate_fn to handle potential None samples
    )
    if len(train_loader) == 0:
        print("ERROR: DataLoader is empty. Check dataset path and filtering.")
        return
    print(f"DataLoader ready with {len(train_loader)} batches.")

    # --- Optimizer and Scheduler ---
    print("Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.scheduler_t0, T_mult=1, eta_min=args.scheduler_eta_min)

    # --- Load Optimizer/Scheduler State (if resuming) ---
    start_epoch = 1
    global_step = 0
    if args.checkpoint_path:
        try:
            optim_scheduler_path = os.path.join(args.checkpoint_path, 'optimizer.pt')
            if os.path.exists(optim_scheduler_path):
                 checkpoint = torch.load(optim_scheduler_path, map_location=args.device)
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                 start_epoch = checkpoint['epoch'] + 1
                 global_step = checkpoint['global_step']
                 print(f"Loaded optimizer and scheduler state from epoch {start_epoch - 1}, global step {global_step}.")
            else:
                 print("Optimizer/Scheduler checkpoint not found. Initializing fresh.")
        except Exception as e:
            print(f"Warning: Could not load optimizer/scheduler state: {e}. Initializing fresh.")
            start_epoch = 1 # Default back if loading fails
            global_step = 0

    # --- WandB Setup ---
    if args.wandb_project:
        print("Initializing WandB...")
        try:
           wandb.login(key=args.wandb_key) # Use key from args or environment
           # Resume if checkpoint provided and wandb_id exists, otherwise start new run
           resume_status = "allow" if args.checkpoint_path else None
           wandb.init(
               project=args.wandb_project,
               name=args.wandb_run_name,
               id=args.wandb_id, # If None, generates a new ID
               resume=resume_status,
               config={
                   "learning_rate": args.learning_rate,
                   "epochs": args.epochs,
                   "batch_size": args.batch_size,
                   "accumulation_steps": args.accumulation_steps,
                   "model_config": model.config.to_dict(),
                   "scheduler_t0": args.scheduler_t0,
                   "scheduler_eta_min": args.scheduler_eta_min
               }
           )
           print(f"WandB initialized. Project: {args.wandb_project}, Run Name: {args.wandb_run_name}, ID: {wandb.run.id}")
        except Exception as e:
            print(f"Error initializing WandB: {e}. WandB logging disabled.")
            args.wandb_project = None # Disable logging if init fails
    else:
        print("WandB project not specified. Logging disabled.")

    # --- Training Loop ---
    print(f"Starting training from epoch {start_epoch}...")
    model.train()
    optimizer.zero_grad() # Ensure grads are zeroed before starting

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        processed_batches = 0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", unit="batch")

        for i, batch in enumerate(batch_iterator):
            # Skip if collate_fn returned None (e.g., all samples in batch failed SNAC)
            if batch is None:
                print(f"Skipping empty batch {i+1}/{len(train_loader)}")
                continue

            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            # text_attention_mask is used for weighted loss calculation
            text_attention_mask = batch['text_attention_mask'].to(args.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits

            # Calculate loss - shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous() # Mask for valid tokens in loss
            shift_text_mask = text_attention_mask[..., 1:].contiguous() # Mask for text tokens in loss

            # Loss function - CrossEntropyLoss, reduce='none' to apply mask & weights
            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Apply weights: lower weight for text tokens
            # Create weights tensor based on text mask
            weights = torch.where(shift_text_mask.view(-1) == 1, args.text_loss_weight, 1.0)
            loss = loss * weights

            # Masked mean loss: only consider non-padded tokens
            # Sum the loss for valid tokens and divide by the number of valid tokens
            valid_token_mask = shift_attention_mask.view(-1) == 1
            masked_loss = torch.sum(loss[valid_token_mask]) / (torch.sum(valid_token_mask) + 1e-8) # Add epsilon for stability

            # Scale loss for gradient accumulation
            scaled_loss = masked_loss / args.accumulation_steps
            scaled_loss.backward()

            # Accumulate gradients
            if (i + 1) % args.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                # Optimizer step
                optimizer.step()
                # Scheduler step (adjust LR)
                scheduler.step()
                # Zero gradients
                optimizer.zero_grad()

                global_step += 1
                current_loss = masked_loss.item() # Log the unscaled loss for this step
                epoch_loss += current_loss
                processed_batches += 1

                # Logging
                lr_rate = scheduler.get_last_lr()[0]
                batch_iterator.set_postfix({
                    "Loss": f"{current_loss:.4f}",
                    "LR": f"{lr_rate:.2e}",
                    "Step": global_step
                    })

                if args.wandb_project:
                    wandb.log({
                        "Training Loss (Step)": current_loss,
                        "Learning Rate": lr_rate,
                        "Global Step": global_step,
                        "Epoch": epoch
                    })

            # Clean up GPU memory
            del input_ids, attention_mask, text_attention_mask, outputs, logits, loss, scaled_loss, masked_loss
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            # Optional: Add a small sleep to prevent potential overloading issues
            # time.sleep(0.01)

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / processed_batches if processed_batches > 0 else 0.0
        print(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.4f}")
        if args.wandb_project:
             wandb.log({"Training Loss (Epoch Avg)": avg_epoch_loss, "Epoch": epoch})

        # --- Save Checkpoint ---
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Saving checkpoint for epoch {epoch} to {checkpoint_dir}...")

            # Save model, tokenizer, and config
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            # Also save generation config if needed (though often loaded with model)
            # generation_config.save_pretrained(checkpoint_dir)

            # Save optimizer and scheduler state
            optim_scheduler_state = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'loss': avg_epoch_loss # Save average epoch loss for reference
            }
            torch.save(optim_scheduler_state, os.path.join(checkpoint_dir, 'optimizer.pt'))
            print(f"Checkpoint saved successfully.")

    # --- End of Training ---
    print("Training finished.")
    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Voice TTS Model")

    # Paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to the preprocessed Hugging Face dataset directory (e.g., 'picked_data').")
    parser.add_argument("--tokenizer_path", type=str, default=".", help="Path to the tokenizer directory (containing tokenizer.json, etc.). Defaults to current directory.")
    parser.add_argument("--config_path", type=str, default=".", help="Path to the model config directory (containing config.json). Defaults to current directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a checkpoint directory to resume training.")
    parser.add_argument("--snac_model_path", type=str, default="hubertsiuzdak/snac_24khz", help="Path or HF ID of the SNAC model.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device.")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="Number of steps to accumulate gradients.")
    parser.add_argument("--text_loss_weight", type=float, default=0.1, help="Weight multiplier for text token loss.")
    parser.add_argument("--grad_clip_norm", type=float, default=5.0, help="Gradient clipping norm value.")

    # Scheduler Hyperparameters
    parser.add_argument("--scheduler_t0", type=int, default=500, help="CosineAnnealingWarmRestarts: Number of steps for the first restart.")
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-7, help="CosineAnnealingWarmRestarts: Minimum learning rate.")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs.")

    # WandB Logging
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name. If None, logging is disabled.")
    parser.add_argument("--wandb_run_name", type=str, default="speech-voice-tts-run", help="WandB run name.")
    parser.add_argument("--wandb_id", type=str, default=None, help="WandB run ID to resume a specific run.")
    parser.add_argument("--wandb_key", type=str, default=os.environ.get("WANDB_API_KEY"), help="WandB API key (can also be set via environment variable).")

    args = parser.parse_args()

    # Calculate effective batch size for logging
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"--- Training Configuration ---")
    print(f"Data Path: {args.data_path}")
    print(f"Tokenizer Path: {args.tokenizer_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Resume Checkpoint: {args.checkpoint_path if args.checkpoint_path else 'None (starting fresh)'}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Accumulation Steps: {args.accumulation_steps}")
    print(f"Effective Batch Size: {effective_batch_size}")
    print(f"Text Loss Weight: {args.text_loss_weight}")
    print(f"Device: {args.device}")
    print(f"WandB Project: {args.wandb_project if args.wandb_project else 'Disabled'}")
    print(f"-----------------------------")

    train(args)