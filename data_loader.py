import torch
from torch.utils.data import Dataset
from snac import SNAC
from transformers import GPT2Tokenizer
from utils import flatten_tensors_adjusted
from datasets import load_from_disk

class CustomDataSet(Dataset):
    def __init__(self, dataset_path, tokenizer_path, snac_model_path="hubertsiuzdak/snac_24khz", max_length=1024, device='cpu'):
        """
        Args:
            dataset_path (str): Path to the processed Hugging Face dataset saved to disk.
            tokenizer_path (str): Path to the trained tokenizer directory.
            snac_model_path (str): Path or Hugging Face ID for the SNAC model.
            max_length (int): Maximum sequence length for the GPT-2 model.
            device (str): Device to run SNAC model on ('cuda' or 'cpu').
        """
        print(f"Loading dataset from {dataset_path}...")
        # Load the specific split needed, or handle splits externally before passing path
        try:
             self.ds = load_from_disk(dataset_path)
        except FileNotFoundError:
             raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please ensure the preprocessed data exists.")
        print(f"Dataset loaded with {len(self.ds)} samples.")

        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        # Ensure PAD token is correctly set if not done during saving
        if self.tokenizer.pad_token is None:
             self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             print("Added [PAD] token to tokenizer.")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        print(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}, Pad token ID: {self.pad_token_id}")

        print(f"Loading SNAC model from {snac_model_path}...")
        self.snac_model = SNAC.from_pretrained(snac_model_path).to(device).eval()
        print("SNAC model loaded.")

        self.max_length = max_length
        self.device = device # Device for SNAC processing

        # Pre-filter dataset for audio length (optional, can be done before saving)
        # self.ds = self.ds.filter(lambda ex: librosa.get_duration(y=ex['audio']['array'], sr=ex['audio']['sampling_rate']) <= 10.0)
        # print(f"Dataset filtered for audio length <= 10s. New size: {len(self.ds)}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        text = item['text_normalized']
        audio_data = item['audio']
        sampling_rate = audio_data['sampling_rate']

        # Ensure audio is float32 tensor, mono, and on the correct device
        audio_tensor = torch.tensor(audio_data['array'], dtype=torch.float32)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0) # Add channel dim
        if audio_tensor.size(0) > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True) # Make mono

        # Resample if necessary (SNAC expects 24kHz)
        if sampling_rate != 24000:
            # Implement resampling if needed, e.g., using torchaudio.transforms.Resample
             raise ValueError(f"Audio sampling rate is {sampling_rate}, but SNAC expects 24000 Hz. Please resample your data.")

        # Add batch dimension for SNAC and move to device
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device) # Shape: [1, 1, T]

        # Get SNAC codes
        with torch.no_grad():
            try:
                _, codes = self.snac_model(audio_tensor) # codes is a list of tensors
            except Exception as e:
                print(f"Error during SNAC processing for index {idx}: {e}")
                # Return None or raise error? For now, return None to skip sample in collate_fn
                # This might happen for very short audio files
                print(f"Skipping sample {idx} due to SNAC error.")
                return None # Signal to collate_fn to skip this sample

        # Flatten SNAC codes
        audio_codes = flatten_tensors_adjusted(codes)
        if not audio_codes: # Handle case where flatten_tensors failed or returned empty
             print(f"Warning: Could not flatten SNAC codes for index {idx}. Skipping sample.")
             return None

        # Tokenize text
        text_tensor = self.tokenizer(text, add_special_tokens=False) # Don't add BOS/EOS here, handle manually
        text_ids = text_tensor['input_ids']

        # Combine tokens: BOS + text + audio + EOS
        # Using tokenizer.bos_token_id which should be <|endoftext|> (50256)
        # Add the SPACER token (50258) before audio codes start
        output_tokens = [self.tokenizer.bos_token_id] + text_ids + audio_codes + [self.eos_token_id]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(output_tokens)

        # Pad sequence
        padding_length = self.max_length - len(output_tokens)
        if padding_length < 0:
            # Truncate if too long (should ideally not happen with pre-filtering)
            print(f"Warning: Sequence length {len(output_tokens)} exceeds max_length {self.max_length}. Truncating.")
            output_tokens = output_tokens[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            # Ensure the last token is EOS if truncated
            if output_tokens[-1] != self.eos_token_id:
                 output_tokens[-1] = self.eos_token_id # Replace last token with EOS
            padding_length = 0
        else:
            output_tokens = output_tokens + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Create text attention mask (1 for text tokens, 0 otherwise) for weighted loss
        # Length should match output_tokens *before* shifting labels
        text_attention_mask = [0] * self.max_length
        # Mark positions corresponding to text_ids (excluding initial BOS)
        start_text_idx = 1
        end_text_idx = start_text_idx + len(text_ids)
        if end_text_idx <= self.max_length: # Ensure indices are within bounds
             for i in range(start_text_idx, end_text_idx):
                 text_attention_mask[i] = 1

        # Convert to tensors
        output_tokens_tensor = torch.tensor(output_tokens, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        text_attention_mask_tensor = torch.tensor(text_attention_mask, dtype=torch.long)

        return {
            'input_ids': output_tokens_tensor,
            'attention_mask': attention_mask_tensor,
            'text_attention_mask': text_attention_mask_tensor # Used for weighted loss
        }

def collate_fn(batch):
    """ Custom collate function to handle potential None values from dataset. """
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch was problematic

    # Standard padding collation if using batch_size > 1 (requires tokenizer setup)
    # For batch_size=1 as in the original script, just stack if needed or return the single item dict
    # If you implement batch_size > 1, you'd use tokenizer.pad here.
    # For now, assuming batch_size=1 or pre-padded data in __getitem__:

    # Stack tensors from the filtered batch
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    text_attention_mask = torch.stack([item['text_attention_mask'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'text_attention_mask': text_attention_mask
    }