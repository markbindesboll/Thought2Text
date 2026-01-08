import os
import random
import json
import torch
import numpy as np
from tqdm import tqdm

os.environ.setdefault("HF_DISABLE_PROGRESS_BARS", "1")
from datautils import EEGDataset, Splitter
from channelnet.model import ChannelNetModel
from channelnet.config import EEGModelConfig
from args import get_args_for_encoder_training
from loss import MSELoss
from transformers import (
    Trainer,
    TrainingArguments,
    AutoProcessor,
    CLIPVisionModelWithProjection,
)
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


class EEGEncoderTrainer(Trainer):
    def __init__(
        self,
        emb_loss_fn=None,
        cls_loss_fn=None,
        clip_model=None,
        data_loaders=None,
        stage1_mode="encode_only",
        tqdm_enabled=True,
        precomputed_embeddings=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_loss_fn = emb_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.clip_model = clip_model
        self.data_loaders = data_loaders
        self.stage1_mode = stage1_mode
        self.precomputed_embeddings = precomputed_embeddings
        self.metric = (
            evaluate.load("accuracy")
            if self.stage1_mode == "classify_and_encode"
            else None
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = "cpu"
        self.tqdm_enabled = tqdm_enabled

    def compute_loss(self, model, inputs, return_outputs=False):
        self.model.train()
        img_data, eeg, labels, image_ids = inputs
        if self.precomputed_embeddings is not None:
            image_embeddings = self.precomputed_embeddings[image_ids.cpu()].to(eeg.device)
        else:
            image_embeddings = self.clip_model(
                pixel_values=img_data["pixel_values"]
            ).image_embeds
        if self.stage1_mode == "encode_only":
            emb_output = model.encode(eeg)
            emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
            loss = emb_loss
            outputs = emb_output
        else:
            emb_output, cls_output = model(eeg)
            emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
            cls_loss = self.cls_loss_fn(cls_output, labels)
            loss = cls_loss + emb_loss
            outputs = cls_output
        self.device = eeg.device
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        return self.data_loaders["train"]

    def get_eval_dataloader(self, eval_dataset=None):
        return self.data_loaders["val"]

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return self.data_loaders["test"]

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset=None)
        eval_metrics = self._run_split(eval_dataloader, split_name="eval")

        # Do testing so trainer_state logs eval/test on same scale
        test_dataloader = self.get_test_dataloader(test_dataset=None)
        test_metrics = self._run_split(test_dataloader, split_name="test")

        metrics = {**eval_metrics, **test_metrics}

        if self.stage1_mode == "classify_and_encode" and "eval_acc" in eval_metrics:
            metrics["eval_loss"] = -eval_metrics["eval_acc"]
        else:
            metrics["eval_loss"] = eval_metrics.get("eval_loss", 0.0)

        return metrics

    def _run_split(self, dataloader, split_name):
        compute_cls = self.stage1_mode == "classify_and_encode"
        split_loss = 0.0
        batch_count = 0
        all_labels = []
        all_preds = []
        iterator = tqdm(
            dataloader,
            disable=not self.tqdm_enabled,
            dynamic_ncols=True,
            leave=False,
        )
        for batch in iterator:
            image_raw, eeg_data, labels, image_ids = batch
            image_raw = image_raw.to(self.device)
            eeg_data = eeg_data.to(self.device)
            if compute_cls:
                labels = labels.to(self.device)
            with torch.no_grad():
                if self.precomputed_embeddings is not None:
                    image_embeddings = self.precomputed_embeddings[image_ids.cpu()].to(self.device)
                else:
                    image_embeddings = self.clip_model(
                        pixel_values=image_raw["pixel_values"]
                    ).image_embeds
                if compute_cls:
                    emb_output, cls_output = self.model(eeg_data)
                    emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
                    cls_loss = self.cls_loss_fn(cls_output, labels)
                    loss = cls_loss + emb_loss
                    preds = self.softmax(cls_output).argmax(dim=1)
                    all_labels.extend(labels.detach().cpu().tolist())
                    all_preds.extend(preds.detach().cpu().tolist())
                else:
                    emb_output = self.model.encode(eeg_data)
                    loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
            split_loss += loss.item()
            batch_count += 1
        avg_loss = split_loss / batch_count if batch_count else 0.0
        metrics = {f"{split_name}_loss": avg_loss}
        if compute_cls and self.metric is not None and all_labels:
            metric = self.metric.compute(
                predictions=all_preds, references=all_labels
            )
            metrics[f"{split_name}_acc"] = metric["accuracy"]
        print(metrics)
        return metrics


def set_gradients(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def main():
    args = get_args_for_encoder_training()
    set_seed(42)
    
    # Load precomputed image embeddings if available
    precomputed_embeddings = None
    embeddings_path = "/zhome/73/b/145313/thesis/data/images/image_embeddings_list.pth"  # Path to precomputed embeddings
    if os.path.exists(embeddings_path):
        print(f"Loading precomputed embeddings from {embeddings_path}")
        precomputed_embeddings = torch.load(embeddings_path, weights_only=False)
        clip_model = None
    else:
        print("No precomputed embeddings found, using CLIP model")
        clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model)
        clip_model.to(args.device)
        clip_model.requires_grad_(False)
        set_gradients(clip_model, False)
        clip_model.eval()

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

    config = EEGModelConfig()

    config.save_pretrained(args.output)
    model = ChannelNetModel(config=config)
    print(f"Stage-1 training mode: {args.stage1_mode}")
    if args.stage1_mode == "encode_only":
        set_gradients(model.classifier, False)

    tqdm_enabled = False

    training_arguments = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        optim=args.optim,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        load_best_model_at_end=True,
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        eval_strategy="epoch",
        report_to="none",
        disable_tqdm=not tqdm_enabled,
    )
    trainer = EEGEncoderTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=dataset,
        emb_loss_fn=MSELoss(),
        cls_loss_fn=torch.nn.CrossEntropyLoss(),
        data_loaders=loaders,
        clip_model=clip_model,
        stage1_mode=args.stage1_mode,
        tqdm_enabled=tqdm_enabled,
        precomputed_embeddings=precomputed_embeddings,
    )
    trainer.train()
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()
