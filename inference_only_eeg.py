# EEG-based image captioning inference with top-1 EEG selection

import random
import logging
import torch
import json
import os
import numpy as np
import warnings

# Suppress repetitive warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from tqdm import tqdm
from args import get_args_for_llm_inference
from model import EEGModelForCausalLM
from datautils import EEGInferenceDataset, SplitterInference
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    args = get_args_for_llm_inference()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if "gemma" in args.model_path.lower():
        messages = [
                {"role": "user", "content": f"<image> Describe this image in one sentence:"},
            ]
    else:
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"<image> Describe this image in one sentence:"},
            ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    max_len = 40  # Reduced from 100 - one sentence doesn't need 100 tokens

    print("Loading model...")
    
    # Load encoder path
    if args.eeg_encoder_path:
        eeg_encoder_path = args.eeg_encoder_path
    else:
        with open(os.path.join(args.model_path, "training_config.json")) as f:
            eeg_encoder_path = json.load(f)["eeg_encoder_path"]
    
    projector_path = os.path.join(args.model_path, "projector.pth")
    
    print(f"\n{'='*60}")
    print(f"Loading Model Components:")
    print(f"  EEG Encoder:  {eeg_encoder_path}")
    print(f"  Projector:    {projector_path}")
    print(f"  LLM:          {args.llm_backbone_name_or_path}")
    print(f"{'='*60}\n")
    
    model = EEGModelForCausalLM.from_separate_pretrained(
        eeg_encoder_path=eeg_encoder_path,
        llm_path=args.llm_backbone_name_or_path,
        use_lora=False,
        llm_low_cpu_mem_usage=True,
    )
    
    model.eeg_encoder.to(args.device)
    model.mm_proj.to(args.device)
    model.eval()  # LLM device handled by accelerate
    
    # Set pad_token_id in generation config to suppress warnings
    model.llm.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Log actual device placement
    logger.info(f"EEG encoder device: {next(model.eeg_encoder.parameters()).device}")
    logger.info(f"Projector device: {next(model.mm_proj.parameters()).device}")
    logger.info(f"LLM device: {next(model.llm.parameters()).device}")

    # Load precomputed image embeddings
    embeddings_path = "/zhome/73/b/145313/thesis/data/images/image_embeddings_list.pth"
    logger.info(f"Loading precomputed image embeddings from {embeddings_path}")
    precomputed_embeddings = torch.load(embeddings_path, weights_only=False)

    # Load precomputed captions (required)
    captions_path = "/zhome/73/b/145313/thesis/data/images/captions_list.pth"  # User must change this
    logger.info(f"Loading ground truth captions from {captions_path}")
    captions = torch.load(captions_path, weights_only=False)

    dataset = EEGInferenceDataset(args=args, captions=captions)
    test_dataloader = DataLoader(
        SplitterInference(
            dataset,
            split_path=args.splits_path,
            split_num=args.split_num,
            split_name="test",
        ),
        batch_size=1,
        drop_last=True,
        shuffle=False,  # Don't shuffle for reproducibility
    )

    # Prepare text prompt tokens (used by both Stage 2 and Stage 3)
    ps = text.split("<image>")
    prefix_ids = tokenizer(ps[0], add_special_tokens=False, truncation=True, return_tensors="pt").input_ids.to(args.device)
    suffix_ids = tokenizer(ps[1].strip(), add_special_tokens=False, truncation=True, return_tensors="pt").input_ids.to(args.device)

    # ========================================================================
    # PHASE 1: Extract embeddings + Select top-1 EEG per image
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Extracting embeddings and selecting top-1 EEG per image...")
    print("="*60)
    
    # Dictionary to store all data grouped by image_id
    image_groups = {}
    
    for batch in tqdm(test_dataloader, desc="Extracting embeddings"):
        eeg, label_string, caption_raw, image_path, image_id = batch
        eeg = eeg.to(args.device)
        
        # Encode EEG to get embeddings
        with torch.no_grad():
            eeg_embeddings = model.eeg_encoder.encode(eeg)
        
        # Get the actual image_id value
        img_id = image_id[0].item() if torch.is_tensor(image_id[0]) else image_id[0]
        
        # Get image embedding
        image_embedding = precomputed_embeddings[image_id].to(args.device)
        
        # Group by image_id
        if img_id not in image_groups:
            image_groups[img_id] = {
                'eeg_embeddings': [],
                'image_embedding': image_embedding,
                'label_string': label_string[0],
                'caption_raw': caption_raw[0],
                'image_path': image_path[0]
            }
        
        image_groups[img_id]['eeg_embeddings'].append(eeg_embeddings)
    
    print(f"Total unique images: {len(image_groups)}")
    print(f"EEG repetitions per image (sample): {len(image_groups[list(image_groups.keys())[0]]['eeg_embeddings'])}")
    
    # Select top-1 EEG embedding per image using cosine similarity
    selected_data = []
    
    for img_id, data in tqdm(image_groups.items(), desc="Top-1 selection"):
        # Stack all EEG embeddings for this image [80, ...]
        eeg_stack = torch.cat(data['eeg_embeddings'], dim=0)
        image_emb = data['image_embedding']
        
        # Normalize for cosine similarity
        eeg_norm = torch.nn.functional.normalize(eeg_stack, p=2, dim=-1)
        img_norm = torch.nn.functional.normalize(image_emb, p=2, dim=-1)
        
        # Compute cosine similarity [80]
        similarities = torch.matmul(eeg_norm, img_norm.T).squeeze()
        
        # Get top-1 index
        top1_idx = torch.argmax(similarities).item()
        top1_similarity = similarities[top1_idx].item()
        
        # Store selected EEG embedding and metadata
        selected_data.append({
            'image_id': img_id,
            'eeg_embedding': data['eeg_embeddings'][top1_idx],
            'image_embedding': image_emb,
            'label_string': data['label_string'],
            'caption_raw': data['caption_raw'],
            'image_path': data['image_path'],
            'similarity_score': top1_similarity,
        })
    
    print(f"Selected {len(selected_data)} EEG embeddings (1 per image)")
    
    # ========================================================================
    # PHASE 2: Generate Stage 2 captions (Image → Stage 2 Projector → LLM)
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Generating Stage 2 captions (Image embeddings)...")
    print("="*60)
    
    # Load Stage 2 projector
    # Extract model name from either HuggingFace ID or cache path
    if "models--" in args.llm_backbone_name_or_path:
        # Cache path format: .../models--org--model/snapshots/hash
        # Extract "model" from "models--org--model"
        parts = args.llm_backbone_name_or_path.split("/")
        model_dir = [p for p in parts if p.startswith("models--")][0]
        llm_name = model_dir.split("--")[-1]  # Get last part after splitting by --
    else:
        # HuggingFace ID format: org/model
        llm_name = args.llm_backbone_name_or_path.split("/")[-1]
    stage2_projector_path = f"/zhome/73/b/145313/Thought2Text/data/runs/{llm_name}/projector.pth"
    logger.info(f"Loading Stage 2 projector from {stage2_projector_path}")
    model.mm_proj.load_state_dict(torch.load(stage2_projector_path, map_location=args.device))
    
    for item in tqdm(selected_data, desc="Stage 2 captions", position=0, leave=True, dynamic_ncols=True):
        with torch.no_grad():
            # Compute and store projected image embedding (before projector swap)
            projected_image_emb = model.mm_proj(item['image_embedding'])
            item['projected_image_embedding'] = projected_image_emb
            
            output_ids_stage2, _ = model.generate(
                input_ids1=prefix_ids,
                input_ids2=suffix_ids,
                mm_embeds=item['image_embedding'],
                max_new_tokens=max_len,
                repetition_penalty=1.1
            )
        caption_stage2 = tokenizer.batch_decode(output_ids_stage2, skip_special_tokens=True)[0]
        item['caption_stage2'] = caption_stage2
    
    # ========================================================================
    # PHASE 3: Generate Stage 3 captions (EEG → Stage 3 Projector → LLM)
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 3: Generating Stage 3 captions (EEG embeddings)...")
    print("="*60)
    
    # Swap to Stage 3 projector
    stage3_projector_path = os.path.join(args.model_path, "projector.pth")
    model.mm_proj.load_state_dict(torch.load(stage3_projector_path, map_location=args.device))
    
    for item in tqdm(selected_data, desc="Stage 3 captions", position=0, leave=True, dynamic_ncols=True):
        with torch.no_grad():
            # Compute projected EEG embedding (with Stage 3 projector loaded)
            projected_eeg_emb = model.mm_proj(item['eeg_embedding'])
            
            # Compute cosine similarity between projected embeddings
            proj_img_norm = F.normalize(item['projected_image_embedding'], p=2, dim=-1)
            proj_eeg_norm = F.normalize(projected_eeg_emb, p=2, dim=-1)
            projected_cosine = F.cosine_similarity(proj_img_norm, proj_eeg_norm, dim=-1)
            item['projected_embedding_similarity'] = projected_cosine.item()
            
            output_ids_stage3, _ = model.generate(
                input_ids1=prefix_ids,
                input_ids2=suffix_ids,
                mm_embeds=item['eeg_embedding'],
                max_new_tokens=max_len,
                repetition_penalty=1.1
            )
        caption_stage3 = tokenizer.batch_decode(output_ids_stage3, skip_special_tokens=True)[0]
        item['caption_stage3'] = caption_stage3
    
    # ========================================================================
    # PHASE 4: Compute caption semantic similarity using LLM embeddings
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 4: Computing caption semantic similarity...")
    print("="*60)
    
    for item in tqdm(selected_data, desc="Caption embeddings", position=0, leave=True, dynamic_ncols=True):
        with torch.no_grad():
            # Tokenize both captions without chat template (raw text only)
            caption2_tokens = tokenizer(item['caption_stage2'], return_tensors="pt", padding=False, add_special_tokens=False)
            caption3_tokens = tokenizer(item['caption_stage3'], return_tensors="pt", padding=False, add_special_tokens=False)
            
            caption2_ids = caption2_tokens.input_ids.to(args.device)
            caption3_ids = caption3_tokens.input_ids.to(args.device)
            
            # Get LLM hidden states for both captions
            outputs2 = model.llm(input_ids=caption2_ids, output_hidden_states=True)
            outputs3 = model.llm(input_ids=caption3_ids, output_hidden_states=True)
            
            # Extract last hidden state [batch, seq_len, hidden_dim]
            hidden2 = outputs2.hidden_states[-1]  # Last layer
            hidden3 = outputs3.hidden_states[-1]
            
            # Use EOS / last non-pad token pooling (following model.py attention mask logic)
            # Since we used padding=False, the last token is always valid
            emb2 = hidden2[0, -1, :]  # [hidden_dim]
            emb3 = hidden3[0, -1, :]  # [hidden_dim]
            
            # Compute cosine similarity between caption embeddings
            caption_cosine = F.cosine_similarity(emb2.unsqueeze(0), emb3.unsqueeze(0), dim=-1)
            item['caption_semantic_similarity'] = caption_cosine.item()
    
    # ========================================================================
    # PHASE 5: Save results to CSV
    # ========================================================================
    print("\n" + "="*60)
    print("PHASE 5: Saving results to CSV...")
    print("="*60)
    
    all_data = []
    for item in selected_data:
        data = {
            "Image ID": item['image_id'],
            "Ground Truth Image": item['image_path'],
            "Expected object": item['label_string'],
            "Expected Caption": item['caption_raw'].replace("<s>", "").replace("</s>", ""),
            "Stage 2 Caption (Image)": item['caption_stage2'],
            "Stage 3 Caption (EEG)": item['caption_stage3'],
            "Cosine": item['similarity_score'],
            "Projected Cosine": item['projected_embedding_similarity'],
            "LLM EOS cosine": item['caption_semantic_similarity'],
        }
        all_data.append(data)
    
    df = pd.DataFrame(all_data)
    df.to_csv(args.dest, index=False)
    print(f"Results saved to: {args.dest}")
    print(f"Total rows: {len(df)}")
    print("="*60)


if __name__ == "__main__":
    main()
