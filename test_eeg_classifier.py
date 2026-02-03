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
        
        # Normalize both modalities (without centering)
        eeg_norm = eeg_raw_avg / (eeg_raw_avg.norm(p=2, dim=1, keepdim=True) + 1e-8)
        img_norm = img_raw_avg / (img_raw_avg.norm(p=2, dim=1, keepdim=True) + 1e-8)
        
        # Compute diagonal metrics (matching image pairs)
        avg_cosines = F.cosine_similarity(eeg_norm, img_norm, dim=1).numpy()
        avg_mses = F.mse_loss(eeg_norm, img_norm, reduction='none').mean(dim=1).numpy()
        
        # Save diagonal metrics to the checkpoint directory
        save_dir = args.output + '/results'
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "avg_cosines.npy"), avg_cosines)
        np.save(os.path.join(save_dir, "avg_mses.npy"), avg_mses)
        print(f"\nSaved avg_cosines.npy and avg_mses.npy to {save_dir}")
        
        # Print overall statistics
        print(f"\nOverall statistics across {len(unique_image_ids)} unique images:")
        print(f"  Average Cosine Similarity (Normalized): {avg_cosines.mean():.4f} ± {avg_cosines.std():.4f}")
        print(f"  Average MSE (Normalized): {avg_mses.mean():.4f} ± {avg_mses.std():.4f}")
        
        # Compute full 200x200 similarity matrices
        print("\nComputing 200x200 matrices...")
        cosine_matrix = torch.mm(eeg_norm, img_norm.t()).numpy()  # [200, 200]
        
        # Compute MSE matrix efficiently using broadcasting (with normalized embeddings)
        eeg_expanded = eeg_norm.unsqueeze(1)  # [200, 1, dim]
        img_expanded = img_norm.unsqueeze(0)  # [1, 200, dim]
        mse_matrix = ((eeg_expanded - img_expanded) ** 2).mean(dim=2).numpy()  # [200, 200]
        
        # Create 200x200 heatmaps
        print("\nCreating visualizations...")
        plt.figure(figsize=(12, 10))
        sns.heatmap(cosine_matrix, annot=False, cmap='RdYlGn', center=0, 
                    cbar_kws={'label': 'Cosine Similarity (Normalized)'}, square=True)
        plt.title(f'Cosine Similarity Heatmap - Normalized Embeddings\n(Averaged across {len(eeg_raw_per_image[unique_image_ids[0]])} repetitions per image)')
        plt.xlabel('CLIP Target Embeddings (Normalized)')
        plt.ylabel('EEG Trained Embeddings (Normalized)')
        plt.tight_layout()
        cosine_heatmap_path = os.path.join(save_dir, "cosine_similarity_heatmap.png")
        plt.savefig(cosine_heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Saved cosine similarity heatmap to {cosine_heatmap_path}")
        plt.close()
        
        # MSE heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(mse_matrix, annot=False, cmap='RdYlGn_r', 
                    cbar_kws={'label': 'MSE (Normalized, lower is better)'}, square=True)
        plt.title(f'MSE Heatmap - Normalized Embeddings\n(Averaged across {len(eeg_raw_per_image[unique_image_ids[0]])} repetitions per image)')
        plt.xlabel('CLIP Target Embeddings (Normalized)')
        plt.ylabel('EEG Trained Embeddings (Normalized)')
        plt.tight_layout()
        mse_heatmap_path = os.path.join(save_dir, "mse_heatmap.png")
        plt.savefig(mse_heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Saved MSE heatmap to {mse_heatmap_path}")
        plt.close()
        
        # Create distribution plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(cosine_matrix.diagonal(), bins=30, edgecolor='black')
        axes[0].set_xlabel('Cosine Similarity (Normalized, Diagonal)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of Cosine Similarity - Normalized\n(n={len(unique_image_ids)} images)')
        axes[0].axvline(cosine_matrix.diagonal().mean(), color='red', linestyle='--', 
                       label=f'Mean: {cosine_matrix.diagonal().mean():.4f}')
        axes[0].legend()
        
        axes[1].hist(mse_matrix.diagonal(), bins=30, edgecolor='black')
        axes[1].set_xlabel('MSE (Normalized, Diagonal)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Distribution of MSE - Normalized\n(n={len(unique_image_ids)} images)')
        axes[1].axvline(mse_matrix.diagonal().mean(), color='red', linestyle='--', 
                       label=f'Mean: {mse_matrix.diagonal().mean():.4f}')
        axes[1].legend()
        
        plt.tight_layout()
        dist_path = os.path.join(save_dir, "metrics_distributions.png")
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics distributions to {dist_path}")
        plt.close()
        
        # Create categorized heatmap
        print("\nCreating categorized heatmap...")
        categories_path = "/zhome/73/b/145313/thesis/data/images/test_super_cat_list.pth"
        categories = torch.load(categories_path, map_location="cpu")
        
        # Sort by category
        sorted_indices = sorted(range(len(categories)), key=lambda i: (categories[i], i))
        sorted_categories = [categories[i] for i in sorted_indices]
        
        # Reorder matrices
        cosine_sorted = cosine_matrix[sorted_indices, :][:, sorted_indices]
        
        # Find category boundaries
        boundaries = []
        category_centers = []
        current_cat = sorted_categories[0]
        start_idx = 0
        
        for i in range(1, len(sorted_categories)):
            if sorted_categories[i] != current_cat:
                boundaries.append(i)
                category_centers.append((start_idx + i) / 2)
                start_idx = i
                current_cat = sorted_categories[i]
        category_centers.append((start_idx + len(sorted_categories)) / 2)
        
        # Get unique categories in order
        unique_cats = []
        for cat in sorted_categories:
            if cat not in unique_cats:
                unique_cats.append(cat)
        
        # Create categorized heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cosine_sorted, annot=False, cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Cosine Similarity (Normalized)'}, 
                   square=True, ax=ax)
        
        # Add category separators
        for boundary in boundaries:
            ax.axhline(boundary, color='black', linewidth=2)
            ax.axvline(boundary, color='black', linewidth=2)
        
        # Set category labels
        ax.set_yticks(category_centers)
        ax.set_yticklabels(unique_cats, rotation=0)
        ax.set_xticks(category_centers)
        ax.set_xticklabels(unique_cats, rotation=90)
        
        ax.set_ylabel('EEG Trained Embeddings')
        ax.set_xlabel('CLIP Target Embeddings')
        plt.title(f'Categorized Cosine Similarity Heatmap\n(Averaged across {len(eeg_raw_per_image[unique_image_ids[0]])} repetitions per image)')
        plt.tight_layout()
        
        categorized_path = os.path.join(save_dir, "categorized_cosine_heatmap.png")
        plt.savefig(categorized_path, dpi=300, bbox_inches='tight')
        print(f"Saved categorized heatmap to {categorized_path}")
        plt.close()


if __name__ == "__main__":
    main()
