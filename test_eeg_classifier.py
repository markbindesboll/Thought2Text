import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from datautils import EEGDataset, Splitter
from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from args import get_args_for_encoder_training
from torch.utils.data import DataLoader, Dataset
import evaluate


def set_seed(seed):
    """Set seed for reproducibility"""
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable to ensure reproducibility


def set_gradients(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def main():
    args = get_args_for_encoder_training()
    set_seed(42)

    # Load precomputed image embeddings if available
    precomputed_embeddings = None
    embeddings_path = "/zhome/73/b/145313/thesis/data/images/image_embeddings_list.pth"
    if os.path.exists(embeddings_path):
        print(f"Loading precomputed embeddings from {embeddings_path}")
        precomputed_embeddings = torch.load(embeddings_path, map_location="cpu")
        if isinstance(precomputed_embeddings, np.ndarray):
            precomputed_embeddings = torch.from_numpy(precomputed_embeddings)
        print(f"Loaded precomputed embeddings shape: {precomputed_embeddings.shape}")

    dataset = EEGDataset(args=args)
    loaders = {
        split: DataLoader(
            Splitter(
                dataset,
                split_path=args.splits_path,
                split_num=args.split_num,
                split_name=split,
            ),
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
        )
        for split in ["train", "val", "test"]
    }
    test_loader = loaders["test"]

    config = EEGModelConfig()

    model = ChannelNetModel.from_pretrained(
        pretrained_model_name_or_path=args.output, config=config
    )
    model.to(args.device)
    model.eval()

    # Check if we're in encode_only mode
    is_encode_only = args.stage1_mode == "encode_only"

    if is_encode_only and precomputed_embeddings is not None:
        # Encode-only mode: compute cosine similarity with precomputed embeddings
        print("Running in encode_only mode with precomputed embeddings")
        sum_cos = 0.0
        count = 0
        
        for batch in tqdm(test_loader, desc="Computing cosine similarities"):
            image_raw, eeg_data, labels, image_ids = batch
            eeg_data = eeg_data.to(args.device)
            
            with torch.no_grad():
                # Get model prediction embedding
                emb_output = model.encode(eeg_data)
                
                # Normalize predicted embeddings
                emb_norm = emb_output / (emb_output.norm(p=2, dim=1, keepdim=True) + 1e-8)
                
                # Index precomputed embeddings by image_ids
                idx = image_ids
                if not torch.is_tensor(idx):
                    idx = torch.as_tensor(idx, dtype=torch.long)
                img_emb_batch = precomputed_embeddings[idx].to(args.device)
                
                # Normalize image embeddings
                img_norm = img_emb_batch / (img_emb_batch.norm(p=2, dim=1, keepdim=True) + 1e-8)
                
                # Compute cosine similarity per sample
                cos = F.cosine_similarity(emb_norm, img_norm, dim=1)
                sum_cos += cos.sum().item()
                count += cos.size(0)
        
        avg_cosine = sum_cos / count if count else 0.0
        print({"test_avg_cosine": avg_cosine})
    else:
        # Classification mode: compute accuracy
        print("Running in classification mode")
        metric = evaluate.load("accuracy")
        softmax = torch.nn.Softmax(dim=1)
        all_labels = []
        all_preds = []
        
        for batch in tqdm(test_loader, desc="Computing predictions"):
            # Try unpacking with image_ids first, fallback to 3-tuple
            if len(batch) == 4:
                image_raw, eeg_data, labels, image_ids = batch
            else:
                image_raw, eeg_data, labels = batch
            
            image_raw = image_raw.to(args.device)
            eeg_data = eeg_data.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                emb_output, cls_output = model(eeg_data)
                preds = softmax(cls_output).argmax(dim=1)
            
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
        
        test_metric = metric.compute(predictions=all_preds, references=all_labels)
        print({"test_acc": test_metric["accuracy"]})


if __name__ == "__main__":
    main()
