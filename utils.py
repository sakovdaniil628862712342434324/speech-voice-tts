import torch

def flatten_tensors_adjusted(tensors):
    """Safely flattens a list of SNAC tensors into a flat list of integers."""
    flattened_list = []
    spacer_token_id = 50258 # Assuming SPACER ID is 50258

    if not tensors or not all(isinstance(t, torch.Tensor) for t in tensors):
        print("Warning: Invalid input tensors for flattening.")
        return flattened_list

    # Ensure tensors are on CPU and are integer type for item()
    tensors = [t.cpu().long() for t in tensors]

    if len(tensors)==3:
      # Expected shapes: [1, N], [1, 2N], [1, 4N]
      if not (tensors[0].ndim == 2 and tensors[1].ndim == 2 and tensors[2].ndim == 2 and
              tensors[0].size(0) == 1 and tensors[1].size(0) == 1 and tensors[2].size(0) == 1 and
              tensors[1].size(1) == 2 * tensors[0].size(1) and
              tensors[2].size(1) == 2 * tensors[1].size(1)):
          print(f"Warning: Unexpected tensor shapes for 3-tensor flattening: {[t.shape for t in tensors]}")
          return flattened_list

      num_frames = tensors[0].size(1)
      for i in range(num_frames):
        flattened_list.append(spacer_token_id)
        flattened_list.append(tensors[0][0, i].item())
        for j in range(2): # Level 2
          flattened_list.append(tensors[1][0, j + i*2].item())
          for k in range(2): # Level 3
            flattened_list.append(tensors[2][0, k + j*2 + i*4].item())

    elif len(tensors)==4:
      # Expected shapes: [1, N], [1, 2N], [1, 4N], [1, 8N]
      if not (tensors[0].ndim == 2 and tensors[1].ndim == 2 and tensors[2].ndim == 2 and tensors[3].ndim == 2 and
              tensors[0].size(0) == 1 and tensors[1].size(0) == 1 and tensors[2].size(0) == 1 and tensors[3].size(0) == 1 and
              tensors[1].size(1) == 2 * tensors[0].size(1) and
              tensors[2].size(1) == 2 * tensors[1].size(1) and
              tensors[3].size(1) == 2 * tensors[2].size(1)):
          print(f"Warning: Unexpected tensor shapes for 4-tensor flattening: {[t.shape for t in tensors]}")
          return flattened_list

      num_frames = tensors[0].size(1)
      for i in range(num_frames): # Level 1
        flattened_list.append(spacer_token_id)
        flattened_list.append(tensors[0][0, i].item())
        for j in range(2): # Level 2
          flattened_list.append(tensors[1][0, j + i*2].item())
          for k in range(2): # Level 3
            flattened_list.append(tensors[2][0, k + j*2 + i*4].item())
            for l in range(2): # Level 4
              flattened_list.append(tensors[3][0, l + k*2 + j*4 + i*8].item())
    else:
        print(f"Warning: Unsupported number of tensors for flattening: {len(tensors)}")

    return flattened_list

def find_last_instance(lst, element):
    """Finds the index of the last occurrence of an element in a list."""
    try:
        return len(lst) - 1 - lst[::-1].index(element)
    except ValueError:
        return -1 # Return -1 if element is not found

def reconstruct_tensors(flattened_output):
    """Reconstructs the list of SNAC tensors from the flattened token sequence."""
    spacer_token_id = 50258
    eos_token_id = 50256
    codes = []
    tensor_levels = [] # Store temp lists for each level

    # 1. Find the start of the audio tokens (first SPACER)
    try:
        start_index = flattened_output.index(spacer_token_id)
    except ValueError:
        print("Warning: No SPACER token found in the output sequence. Cannot reconstruct audio.")
        return []

    # 2. Find the end of the audio tokens (last EOS before potential padding)
    # We look for the *last* EOS token overall, assuming generation stops there.
    end_index = find_last_instance(flattened_output, eos_token_id)
    if end_index == -1 or end_index <= start_index:
       # If no EOS or EOS is before the first spacer, use the whole list after start
       print("Warning: No EOS token found after the first SPACER. Using sequence until the end.")
       audio_tokens = flattened_output[start_index:]
    else:
        audio_tokens = flattened_output[start_index:end_index] # Exclude the EOS token itself

    if not audio_tokens:
        print("Warning: No audio tokens found between SPACER and EOS.")
        return []

    # 3. Determine the number of levels based on the first block
    try:
        next_spacer_index = audio_tokens.index(spacer_token_id, 1) # Find second spacer
        tokens_per_frame = next_spacer_index # Includes the spacer itself
    except ValueError:
        # Only one spacer found, assume it's the only frame
        tokens_per_frame = len(audio_tokens)

    num_levels = 0
    if tokens_per_frame == 1 + 1 + 2 + 4: # Spacer + L1 + L2 + L3
        num_levels = 3
        expected_tokens_per_frame = 8
        print("Reconstructing 3 levels of SNAC codes.")
    elif tokens_per_frame == 1 + 1 + 2 + 4 + 8: # Spacer + L1 + L2 + L3 + L4
        num_levels = 4
        expected_tokens_per_frame = 16
        print("Reconstructing 4 levels of SNAC codes.")
    else:
        print(f"Warning: Unexpected number of tokens per frame block: {tokens_per_frame}. Cannot determine SNAC levels.")
        return []

    # Initialize lists for each level
    for _ in range(num_levels):
        tensor_levels.append([])

    # 4. Iterate through the audio tokens and reconstruct
    for i in range(0, len(audio_tokens), expected_tokens_per_frame):
        frame_block = audio_tokens[i : i + expected_tokens_per_frame]

        if len(frame_block) < expected_tokens_per_frame:
             print(f"Warning: Incomplete frame block at the end. Length: {len(frame_block)}. Skipping.")
             continue
        if frame_block[0] != spacer_token_id:
             print(f"Warning: Expected SPACER token at index {i}, but found {frame_block[0]}. Skipping frame.")
             continue

        current_idx = 1 # Start after spacer
        # Level 1
        tensor_levels[0].append(frame_block[current_idx])
        current_idx += 1
        # Level 2
        tensor_levels[1].extend(frame_block[current_idx : current_idx+2])
        current_idx += 2
        # Level 3
        tensor_levels[2].extend(frame_block[current_idx : current_idx+4])
        current_idx += 4
        # Level 4 (if applicable)
        if num_levels == 4:
            tensor_levels[3].extend(frame_block[current_idx : current_idx+8])
            # current_idx += 8 # No need to update further for this frame

    # 5. Convert lists to tensors
    try:
        codes = [torch.tensor(level).unsqueeze(0).long() for level in tensor_levels] # Shape [1, N*2^k]
        print(f"Reconstructed tensor shapes: {[c.shape for c in codes]}")
    except Exception as e:
        print(f"Error converting reconstructed lists to tensors: {e}")
        return []

    return codes