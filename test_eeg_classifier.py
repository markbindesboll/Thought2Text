import os
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
import matplotlib.pyplot as plt
import seaborn as sns


def set_gradients(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def main():
    args = get_args_for_encoder_training()

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
    test_loader = DataLoader(
        Splitter(
            dataset,
            split_path=args.splits_path,
            split_num=args.split_num,
            split_name="test",
        ),
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
    )

    config = EEGModelConfig()

    model = ChannelNetModel.from_pretrained(
        pretrained_model_name_or_path=args.output, config=config
    )
    model.to(args.device)
    model.eval()

    # Check if we're in encode_only mode
    is_encode_only = args.stage1_mode == "encode_only"

    if is_encode_only and precomputed_embeddings is not None:
        # Encode-only mode: compute cosine similarity and MSE with precomputed embeddings
        print("Running in encode_only mode with precomputed embeddings")
        
        # Single pass: collect EEG embeddings per image (image embeddings will be retrieved later)
        print("Processing test data...")
        total_samples = 0
        eeg_raw_per_image = {}  # image_id -> list of raw eeg embeddings
        
        for batch in tqdm(test_loader, desc="Computing embeddings"):
            image_raw, eeg_data, labels, image_ids = batch
            eeg_data = eeg_data.to(args.device)
            
            with torch.no_grad():
                # Get model prediction embedding (only once!)
                emb_output = model.encode(eeg_data)
                
                total_samples += emb_output.size(0)
                
                # Store EEG embeddings per image (image embeddings will be retrieved from precomputed)
                for i, img_id in enumerate(image_ids.tolist()):
                    if img_id not in eeg_raw_per_image:
                        eeg_raw_per_image[img_id] = []
                    eeg_raw_per_image[img_id].append(emb_output[i].cpu())
        
        print(f"Processed {total_samples} samples")
        
        # Average EEG embeddings across repetitions for each image
        print("\nAveraging embeddings across repetitions...")
        unique_image_ids = sorted(eeg_raw_per_image.keys())
        
        # Print image IDs for inspection
        print(f"Unique image IDs: min={unique_image_ids[0]}, max={unique_image_ids[-1]}, count={len(unique_image_ids)}")
        print(f"Image ID range: {unique_image_ids[0]}-{unique_image_ids[-1]}")
        
        eeg_raw_avg = torch.stack([torch.stack(eeg_raw_per_image[img_id]).mean(dim=0) 
                                     for img_id in unique_image_ids])  # [200, embedding_dim]
        
        # Get image embeddings directly from precomputed using unique_image_ids
        img_raw_avg = precomputed_embeddings[unique_image_ids]  # [200, embedding_dim]
        
        # Compute image space mean (CLIP reference space) from test set
        img_global_mean = img_raw_avg.mean(dim=0)
        
        # Center both modalities using CLIP image space mean (reference space)
        # This preserves any global bias/shift in EEG embeddings for diagnostic purposes
        eeg_centered = eeg_raw_avg - img_global_mean.cpu()
        img_centered = img_raw_avg - img_global_mean.cpu()
        eeg_norm = eeg_centered / (eeg_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
        img_norm = img_centered / (img_centered.norm(p=2, dim=1, keepdim=True) + 1e-8)
        
        # Compute diagonal metrics (matching image pairs)
        avg_cosines = F.cosine_similarity(eeg_norm, img_norm, dim=1).numpy()
        avg_mses = F.mse_loss(eeg_raw_avg, img_raw_avg, reduction='none').mean(dim=1).numpy()
        
        # Save diagonal metrics to the checkpoint directory
        save_dir = args.output + '/results'
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "avg_cosines.npy"), avg_cosines)
        np.save(os.path.join(save_dir, "avg_mses.npy"), avg_mses)
        print(f"\nSaved avg_cosines.npy and avg_mses.npy to {save_dir}")
        
        # Print overall statistics
        print(f"\nOverall statistics across {len(unique_image_ids)} unique images:")
        print(f"  Average Centered Cosine Similarity: {avg_cosines.mean():.4f} ± {avg_cosines.std():.4f}")
        print(f"  Average MSE: {avg_mses.mean():.4f} ± {avg_mses.std():.4f}")
        
        # Compute full 200x200 similarity matrices
        print("\nComputing 200x200 matrices...")
        cosine_matrix = torch.mm(eeg_norm, img_norm.t()).numpy()  # [200, 200]
        
        # Compute MSE matrix efficiently using broadcasting
        eeg_expanded = eeg_raw_avg.unsqueeze(1)  # [200, 1, dim]
        img_expanded = img_raw_avg.unsqueeze(0)  # [1, 200, dim]
        mse_matrix = ((eeg_expanded - img_expanded) ** 2).mean(dim=2).numpy()  # [200, 200]
        
        # Create 200x200 heatmaps
        print("\nCreating visualizations...")
        plt.figure(figsize=(12, 10))
        sns.heatmap(cosine_matrix, annot=False, cmap='RdYlGn', center=0, 
                    cbar_kws={'label': 'Average Cosine Similarity'}, square=True)
        plt.title(f'Average Cosine Heatmap (All Subjects)\n(Averaged across {len(eeg_raw_per_image[unique_image_ids[0]])} repetitions per image)')
        plt.xlabel('CLIP Target Embeddings')
        plt.ylabel('EEG Trained Embeddings')
        plt.tight_layout()
        cosine_heatmap_path = os.path.join(save_dir, "cosine_similarity_heatmap.png")
        plt.savefig(cosine_heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Saved cosine similarity heatmap to {cosine_heatmap_path}")
        plt.close()
        
        # MSE heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(mse_matrix, annot=False, cmap='RdYlGn_r', 
                    cbar_kws={'label': 'MSE (lower is better)'}, square=True)
        plt.title(f'MSE Heatmap\n(Averaged across {len(eeg_raw_per_image[unique_image_ids[0]])} repetitions per image)')
        plt.xlabel('CLIP Target Embeddings')
        plt.ylabel('EEG Trained Embeddings')
        plt.tight_layout()
        mse_heatmap_path = os.path.join(save_dir, "mse_heatmap.png")
        plt.savefig(mse_heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Saved MSE heatmap to {mse_heatmap_path}")
        plt.close()
        
        # Create distribution plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(cosine_matrix.diagonal(), bins=30, edgecolor='black')
        axes[0].set_xlabel('Centered Cosine Similarity (Diagonal)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of Centered Cosine Similarity\n(n={len(unique_image_ids)} images)')
        axes[0].axvline(cosine_matrix.diagonal().mean(), color='red', linestyle='--', 
                       label=f'Mean: {cosine_matrix.diagonal().mean():.4f}')
        axes[0].legend()
        
        axes[1].hist(mse_matrix.diagonal(), bins=30, edgecolor='black')
        axes[1].set_xlabel('MSE (Diagonal)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Distribution of MSE\n(n={len(unique_image_ids)} images)')
        axes[1].axvline(mse_matrix.diagonal().mean(), color='red', linestyle='--', 
                       label=f'Mean: {mse_matrix.diagonal().mean():.4f}')
        axes[1].legend()
        
        plt.tight_layout()
        dist_path = os.path.join(save_dir, "metrics_distributions.png")
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics distributions to {dist_path}")
        plt.close()
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
