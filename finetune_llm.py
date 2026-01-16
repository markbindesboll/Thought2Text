# Program for fine tuning eeg_encoder through image embeddings and contrastive loss
# sample command:

# python finetune_llm.py
#   --eeg_dataset data/block/eeg_55_95_std.pth
#   --splits_path data/block/block_splits_by_image_all.pth
#   --eeg_encoder_path ./eeg_encoder_55-95_40_classes
#   --image_dir data/images/ --output mistral7b-eeg_55_95_40_classes
#   --llm_backbone_name_or_path mistralai/Mistral-7B-Instruct-v0.3
#   --load_in_8bit

# For skipping stage3:

# python finetune_llm.py --eeg_dataset data/block/eeg_55_95_std.pth --splits_path data/block/block_splits_by_image_all.pth --eeg_encoder_path ./eeg_encoder_55-95_40_classes --image_dir data/images/ --output mistral7b-eeg_55_95_40_classes_no_stage3 --llm_backbone_name_or_path mistralai/Mistral-7B-Instruct-v0.3 --load_in_8bit --no_stage3


import os
import gc
import random
import logging
import torch
import numpy as np
import json
import copy
from transformers import (
    CLIPVisionModelWithProjection,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from datautils import (
    EEGFineTuningDataset,
    SplitterFineTuning,
    Filter
)
from torch.utils.data import Dataset, DataLoader
from args import get_args_for_llm_finetuning
from model import EEGModelForCausalLM


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class Stage2Trainer(Trainer):
    def __init__(self, clip_model=None, data_loaders=None, tokenizer=None, precomputed_embeddings=None, **kwargs):
        super().__init__(**kwargs)
        self.clip_model = clip_model
        self.data_loaders = data_loaders
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.precomputed_embeddings = precomputed_embeddings

    def compute_loss(self, model, inputs, return_outputs=False):
        (
            eeg_data,
            input_ids1,
            input_ids2,
            label_string,
            image_ids,
        ) = inputs
        # Use precomputed embeddings indexed by image_id
        if self.precomputed_embeddings is None:
            raise RuntimeError("Precomputed embeddings are required for Stage 2 training")
        image_embeddings = self.precomputed_embeddings[image_ids.cpu()].to(self.device)
        output, labels = model(
            input_ids1=input_ids1, input_ids2=input_ids2, mm_embeds=image_embeddings
        )
        # print("Labels", self.tokenizer.batch_decode(labels))
        return (output.loss, output) if return_outputs else output.loss

    def get_train_dataloader(self):
        return self.data_loaders["train"]

    def get_eval_dataloader(self, eval_dataset=None):
        return self.data_loaders["val"]

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return self.data_loaders["test"]


class Stage3Trainer(Trainer):
    def __init__(self, data_loaders=None, tokenizer=None, use_filter=False, eeg_encoder=None, **kwargs):
        super().__init__(**kwargs)
        self.data_loaders = data_loaders
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.use_filter = use_filter
        self.eeg_encoder = eeg_encoder

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.use_filter:
            # Filtered data: (eeg_embeddings, input_ids1, input_ids2)
            (
                eeg_embeddings,
                input_ids1,
                input_ids2,
            ) = inputs
        else:
            # Unfiltered data: (eeg, input_ids1, input_ids2, label_string, image_id)
            # Need to encode EEG to get embeddings
            (
                eeg,
                input_ids1,
                input_ids2,
                label_string,
                image_id,
            ) = inputs
            with torch.no_grad():
                eeg_embeddings = self.eeg_encoder.encode(eeg)
        
        output, labels = model(
            input_ids1=input_ids1, input_ids2=input_ids2, mm_embeds=eeg_embeddings
        )
        # print("Labels", self.tokenizer.batch_decode(labels))
        return (output.loss, output) if return_outputs else output.loss

    def get_train_dataloader(self):
        return self.data_loaders["train"]

    def get_eval_dataloader(self, eval_dataset=None):
        return self.data_loaders["val"]

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return self.data_loaders["test"]


def main():
    set_seed(42)
    args = get_args_for_llm_finetuning()
    dtype = torch.float32

    if args.load_in_8bit:
        logger.info("Model in INT8")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
        model = EEGModelForCausalLM.from_separate_pretrained(
            eeg_encoder_path=args.eeg_encoder_path,
            llm_path=args.llm_backbone_name_or_path,
            use_lora=args.use_lora,
            llm_quantization_config=quantization_config,
            llm_low_cpu_mem_usage=True,
        )
        args.optim = "paged_adamw_8bit"
        model.eeg_encoder.to(args.device)
        model.mm_proj.to(args.device)

    else:
        logger.info("Model in FULL")
        model = EEGModelForCausalLM.from_separate_pretrained(
            eeg_encoder_path=args.eeg_encoder_path,
            llm_path=args.llm_backbone_name_or_path,
            use_lora=args.use_lora,
            llm_low_cpu_mem_usage=True,
        )
        model.eeg_encoder.to(args.device)
        model.mm_proj.to(args.device)

    model.llm.save_pretrained(os.path.join(args.output, "llm"))
    model.train()
    set_gradients(module=model.eeg_encoder, requires_grad=False)
    set_gradients(module=model.llm, requires_grad=False)

    # Load precomputed image embeddings (required for stage 2)
    embeddings_path = "/zhome/73/b/145313/thesis/data/images/image_embeddings_list.pth"  # User must change this
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Precomputed embeddings not found at {embeddings_path}. "
            "Please provide valid embeddings file for stage 2 training."
        )
    logger.info(f"Loading precomputed embeddings from {embeddings_path}")
    precomputed_embeddings = torch.load(embeddings_path, weights_only=False)
    
    # Load precomputed captions (required)
    captions_path = "/zhome/73/b/145313/thesis/data/images/captions_list.pth"  # User must change this
    if not os.path.exists(captions_path):
        raise FileNotFoundError(
            f"Ground Truth captions not found at {captions_path}. "
            "Please provide valid captions file."
        )
    logger.info(f"Loading ground truth captions from {captions_path}")
    captions = torch.load(captions_path, weights_only=False)

    dataset = EEGFineTuningDataset(
        args=args, tokenizer_path=args.llm_backbone_name_or_path, captions=captions
    )
    
    if not args.no_stage2:
        logger.info("STAGE 2: LLM fine tuning on images")
        llm_name = args.llm_backbone_name_or_path.split("/")[1]
        pretrained_path = os.path.join(args.saved_pretrained_model_path, llm_name)
        llm_path = os.path.join(pretrained_path, "llm")
        projector_path = os.path.join(pretrained_path, "projector.pth")
        
        # Check if Stage 2 model actually exists (not just the directory)
        if os.path.exists(llm_path) and os.path.exists(projector_path):
            print(f"Stage 2 model found at {pretrained_path}. Loading LLM and projector, using encoder from args.")
            del model
            gc.collect()
            # Load Stage 2 model but use the encoder from args (not from checkpoint)
            model = EEGModelForCausalLM.from_separate_pretrained(
                eeg_encoder_path=args.eeg_encoder_path,
                llm_path=llm_path,
                use_lora=args.use_lora,
                llm_low_cpu_mem_usage=True,
            )
            # Load the projector weights from Stage 2
            model.mm_proj.load_state_dict(torch.load(projector_path))
            
            model.eeg_encoder.to(args.device)
            model.mm_proj.to(args.device)
            set_gradients(module=model.eeg_encoder, requires_grad=False)
            model.llm.save_pretrained(llm_path)


        else:           
            training_arguments_stage2 = TrainingArguments(
                output_dir=args.output,
                num_train_epochs=args.num_epochs_image,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_checkpointing=True,
                optim=args.optim,
                save_steps=args.save_steps,
                logging_steps=args.logging_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                fp16=args.fp16,
                bf16=args.bf16,
                max_grad_norm=args.max_grad_norm,
                max_steps=args.max_steps,
                warmup_ratio=args.warmup_ratio,
                group_by_length=args.group_by_length,
                lr_scheduler_type=args.lr_scheduler_type,
                report_to="tensorboard",
            )


            # Stage 2 uses precomputed embeddings (no CLIP model needed)
            loaders = {
                split: DataLoader(
                    SplitterFineTuning(
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

            trainer = Stage2Trainer(
                model=model,
                args=training_arguments_stage2,
                train_dataset=dataset,
                eval_dataset=dataset,
                data_loaders=loaders,
                clip_model=None,
                tokenizer=dataset.tokenizer,
                precomputed_embeddings=precomputed_embeddings,
            )
            trainer.train()
            # Save Stage 2: only LLM and projector (not encoder - it's frozen and comes from args)
            os.makedirs(pretrained_path, exist_ok=True)
            torch.save(model.mm_proj.state_dict(), os.path.join(pretrained_path, "projector.pth"))
            model.llm.save_pretrained(os.path.join(pretrained_path, "llm"))
            dataset.tokenizer.save_pretrained(pretrained_path)

            del loaders
            gc.collect()
    
    # Stage 3: Train on EEG data
    # Check if encoder was trained in encode_only mode (classifier not trained)
    # If so, skip filtering since we can't predict labels
    encoder_config_path = os.path.join(args.eeg_encoder_path, "config.json")
    stage1_mode = "encode_only"  # default
    if os.path.exists(encoder_config_path):
        with open(encoder_config_path) as f:
            encoder_config = json.load(f)
            # Check if this info is stored, otherwise assume encode_only
            stage1_mode = encoder_config.get("stage1_mode", "encode_only")
    
    logger.info(f"Stage 3: EEG encoder was trained in '{stage1_mode}' mode")
    
    if stage1_mode == "encode_only":
        # No filtering - encoder can't predict labels reliably
        logger.info("Skipping data filtering (encode_only mode)")
        loaders = {
            split: DataLoader(
                SplitterFineTuning(
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
    else:
        # Use Filter to keep only samples with correct predicted labels
        logger.info("Applying data filtering (classify_and_encode mode)")
        loaders = {
            split: DataLoader(
                Filter(SplitterFineTuning(
                    dataset,
                    split_path=args.splits_path,
                    split_num=args.split_num,
                    split_name=split,
                ),eeg_encoder = model.eeg_encoder, device = args.device),
                batch_size=args.batch_size,
                drop_last=True,
                shuffle=True,
            )
            for split in ["train", "val", "test"]
        }
    

    training_arguments_stage3 = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.num_epochs_eeg,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="tensorboard",
    )

    trainer = Stage3Trainer(
        model=model,
        args=training_arguments_stage3,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_loaders=loaders,
        tokenizer=dataset.tokenizer,
        use_filter=(stage1_mode != "encode_only"),
        eeg_encoder=model.eeg_encoder,
    )
    trainer.train()
    model.save_pretrained(args.output)
    dataset.tokenizer.save_pretrained(args.output)
    with open(os.path.join(args.output, "id2label.json"), "w") as f:
        json.dump(dataset.id2label, f)


if __name__ == "__main__":
    main()
