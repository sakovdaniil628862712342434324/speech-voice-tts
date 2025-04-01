import torch
import soundfile as sf
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from snac import SNAC
from utils import reconstruct_tensors
import argparse
import os
import time

def infer(args):
    # --- Setup ---
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("WARNING: CUDA not available, switching to CPU for inference.")
        args.device = 'cpu'

    # Ensure output directory exists if specified path includes directories
    output_dir = os.path.dirname(args.output_wav_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --- Load Models ---
    print(f"Loading Speech Voice TTS model from: {args.model_path}")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
        model.to(args.device).eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error loading model or tokenizer from {args.model_path}: {e}")
        return

    print(f"Loading SNAC model from: {args.snac_model_path}")
    try:
        snac_model = SNAC.from_pretrained(args.snac_model_path)
        snac_model.to(args.device).eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error loading SNAC model: {e}")
        return

    print("Models loaded successfully.")

    # --- Prepare Input ---
    print(f"Input text: '{args.input_text}'")
    # Tokenize input text: Add BOS token and the SPACER token manually before generation
    # We need BOS to start generation and SPACER to signal the transition to audio tokens
    input_ids = tokenizer(args.input_text, return_tensors='pt', add_special_tokens=False).input_ids
    spacer_token_id = tokenizer.convert_tokens_to_ids("SPACER") # Get SPACER ID dynamically
    bos_token_id = tokenizer.bos_token_id

    # Prepend BOS and append SPACER
    input_ids = torch.cat([
        torch.tensor([[bos_token_id]]),
        input_ids,
        torch.tensor([[spacer_token_id]])
    ], dim=1).to(args.device)

    # Create a simple attention mask (all 1s for the input prompt)
    attention_mask = torch.ones_like(input_ids).to(args.device)

    print(f"Input token IDs (with BOS and SPACER): {input_ids.tolist()}")

    # --- Generate Audio Tokens ---
    print("Generating audio tokens...")
    start_time = time.time()
    with torch.no_grad():
        # Generate sequence including prompt, audio codes, and EOS
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            num_beams=args.num_beams,
            do_sample=args.do_sample if args.do_sample else True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id, # Important for beam search
            eos_token_id=tokenizer.eos_token_id # Stop generation at EOS
        )
    end_time = time.time()
    print(f"Token generation took {end_time - start_time:.2f} seconds.")
    print(f"Generated sequence length: {output_ids.shape[1]}")

    # Extract generated sequence (remove prompt part if needed, but reconstruction handles it)
    generated_sequence = output_ids[0].tolist()
    # print(f"Generated token sequence (first 100): {generated_sequence[:100]}...")

    # --- Reconstruct SNAC Codes ---
    print("Reconstructing SNAC codes from tokens...")
    reconstructed_codes = reconstruct_tensors(generated_sequence)

    if not reconstructed_codes:
        print("ERROR: Failed to reconstruct SNAC codes. Cannot generate audio.")
        return

    # Move reconstructed codes to the correct device for SNAC decoding
    reconstructed_codes = [code.to(args.device) for code in reconstructed_codes]

    # --- Decode to Audio ---
    print("Decoding SNAC codes to audio waveform...")
    start_time = time.time()
    with torch.no_grad():
        # SNAC decode expects a list of tensors, typically [1, N], [1, 2N], [1, 4N], ...
        audio_hat = snac_model.decode(reconstructed_codes)
    end_time = time.time()
    print(f"SNAC decoding took {end_time - start_time:.2f} seconds.")

    # --- Save Audio ---
    # Output shape is likely [1, 1, T] -> squeeze to [T] for saving
    audio_waveform = audio_hat.squeeze().cpu().numpy()
    sampling_rate = 24000 # SNAC default sample rate
    print(f"Saving generated audio to: {args.output_wav_path}")
    try:
        sf.write(args.output_wav_path, audio_waveform, sampling_rate)
        print("Audio saved successfully.")
    except Exception as e:
        print(f"Error saving audio file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Speech Voice TTS Model")

    # Paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained Speech Voice TTS model directory (containing config, model weights, tokenizer).")
    parser.add_argument("--snac_model_path", type=str, default="hubertsiuzdak/snac_24khz", help="Path or HF ID of the SNAC model.")
    parser.add_argument("--output_wav_path", type=str, required=True, help="Path to save the generated .wav file.")

    # Input
    parser.add_argument("--input_text", type=str, required=True, help="Text prompt to convert to speech.")

    # Generation Parameters
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for generation.")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search.")
    parser.add_argument("--do_sample", action='store_true', help="Use sampling instead of greedy/beam search. If False, beam search is used.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (only used if --do_sample is True).")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling probability (only used if --do_sample is True).")
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help="Penalty for repeating tokens.")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()

    print(f"--- Inference Configuration ---")
    print(f"Model Path: {args.model_path}")
    print(f"Output WAV Path: {args.output_wav_path}")
    print(f"Input Text: '{args.input_text}'")
    print(f"Max Length: {args.max_length}")
    print(f"Num Beams: {args.num_beams}")
    print(f"Do Sample: {args.do_sample}")
    print(f"Temperature: {args.temperature if args.do_sample else 'N/A'}")
    print(f"Top-p: {args.top_p if args.do_sample else 'N/A'}")
    print(f"Repetition Penalty: {args.repetition_penalty}")
    print(f"Device: {args.device}")
    print(f"-----------------------------")

    infer(args)

"""
Example usage:

python infer.py \
    --model_path model \
    --output_wav_path output.wav \
    --input_text "Hello, this is a test."

"""