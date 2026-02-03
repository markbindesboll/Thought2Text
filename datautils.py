import os
import torch
import logging
import tqdm
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor


logger = logging.getLogger(__name__)


class EEGDataset:

    # Constructor
    def __init__(self, args):
        self.args = args
        # Load EEG signals
        loaded = torch.load(args.eeg_dataset, weights_only=False)

        self.data = loaded["dataset"]
        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.channels = loaded["channels"]
        self.times = loaded["times"]

        # Compute size
        self.size = len(self.data)
        self.image_dir = args.image_dir

        # Initialize image processor
        self.processor = AutoProcessor.from_pretrained(args.clip_model)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):

        # Process EEG
        eeg = self.data[i]["eeg"].float()
        #eeg = eeg[self.args.time_low : self.args.time_high, :]
        #eeg = eeg.t()
        eeg = eeg.view(1, len(self.channels), len(self.times))
        label_id = self.data[i]["label"]
        label_string = self.labels[label_id]

        image_id = self.data[i]["image"]
        image_name = self.images[image_id]
        # Sanity check
        # print("n_channels:", len(self.channels), "n_times:", len(self.times))
        # print("times[0], times[-1]:", self.times[0], self.times[-1])
        # print(self.data[i]["eeg"].shape)
        # print("image_name:", image_name)
        # print("ImageID:", self.data[i]["image"])
        # print("label_string:", label_string)
        # print("labelID:", label)
        # print(i)


        if label_id<1654:
            image_path = os.path.join(
                self.image_dir, "training_images",label_string, image_name
            )
        else:
            image_path = os.path.join(
                self.image_dir, "test_images",label_string, image_name
            )
        image_raw = Image.open(image_path).convert("RGB")

        image_raw = self.processor(images=image_raw, return_tensors="pt", padding=True)
        image_raw["pixel_values"] = image_raw["pixel_values"].squeeze(0)

        return image_raw, eeg, label_id, image_id


class Splitter:

    def __init__(
        self, dataset, split_path, split_num=0, split_name="train"
    ):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path, weights_only=False)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Compute size
        self.size = len(self.split_idx)
        print(f"Total examples in the split {split_name} {self.size}")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        # Preserve image_id so downstream code (train/test) can index precomputed embeddings
        img_data, eeg, label, image_id = self.dataset[self.split_idx[i]]
        return img_data, eeg, label, image_id



        

class EEGFineTuningDataset:

    # Constructor
    def __init__(
        self,
        args,
        tokenizer_path=None,
        max_len=512,
        captions=None,
    ):
        
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.max_len = max_len
        self.captions = captions
        if "gemma" in tokenizer_path.lower():
            self.messages = [
                {"role": "user", "content": f"<image> <label_string> Describe this image in one sentence:"},
            ]
            # Gemmas do not have system role
        else:
            self.messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"<image> <label_string> Describe this image in one sentence:"},
            ]
        
        
        # Load EEG signals
        loaded = torch.load(args.eeg_dataset,weights_only=False)
        if args.subject != 0:
            self.data = [
                loaded["dataset"][i]
                for i in range(len(loaded["dataset"]))
                if loaded["dataset"][i]["subject"] == args.subject
            ]
        else:
            self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.channels = loaded["channels"]
        self.times = loaded["times"]

        # Compute size
        self.size = len(self.data)
        self.image_dir = args.image_dir
        self.id2label = {}

        # Initialize image processor
        self.processor = AutoProcessor.from_pretrained(args.clip_model)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):

        # Process EEG
        eeg = self.data[i]["eeg"].float()
        eeg = eeg.view(1, len(self.channels), len(self.times))
        label = self.data[i]["label"]
        label_string = self.labels[label]
        image_id = self.data[i]["image"]
        
        self.id2label[label] = label_string
        
        # Load caption from precomputed list (required)
        if self.captions is None:
            raise RuntimeError(
                "Captions are required but not provided. "
                "Please provide precomputed captions indexed by image_id."
            )
        content = self.captions[image_id]
        
        message = self.messages+[{"role": "assistant", "content" : content}]
        
        text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        # Strip numeric prefix (e.g., "0001_Aardvark" -> "Aardvark")
        clean_label = label_string.split('_', 1)[1] if '_' in label_string else label_string
        new_text = text.replace("<label_string>", clean_label)
        ps = new_text.split("<image>")
        prefix = ps[0]
        suffix = ps[1]
        #print (suffix)
        input_ids1 = self.tokenizer(
            prefix,
            padding="max_length",
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        input_ids2 = self.tokenizer(
            suffix,
            padding="max_length",
            add_special_tokens=False,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        input_ids1 = input_ids1.squeeze(0)
        input_ids2 = input_ids2.squeeze(0)

        return eeg, input_ids1, input_ids2, label_string, image_id


class SplitterFineTuning:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # f.dataset = dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path,weights_only=False)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Compute size
        self.size = len(self.split_idx)
        print(f"Total examples in the split {split_name} {self.size}")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, input_ids1, input_ids2, label_string, image_id = self.dataset[self.split_idx[i]]
        return eeg, input_ids1, input_ids2, label_string, image_id


class Filter:
    # this is to filter datapoints which have valid predicted object labels
    def __init__(self, dataset, eeg_encoder, device = "cpu") -> None:
        dl = DataLoader(dataset=dataset, batch_size=128, shuffle=False)
        self.data = []

        for batch in tqdm.tqdm(dl):
            _, eeg, input_ids1, input_ids2, label_string, image_id = batch
            
            eeg = eeg.to(device)
            with torch.no_grad():
                mm_embeds, cls_logits = eeg_encoder(eeg)
            obj_labels = F.softmax(cls_logits, dim=1).argmax(dim=1)
            for i, ls in enumerate(label_string):
                mm_embeds_i = mm_embeds[i]
                input_ids1_i = input_ids1[i]
                input_ids2_i = input_ids2[i]
                label_string_predicted_i = id2label[str(obj_labels[i].item())]
                if label_string_predicted_i == ls:
                    self.data.append([mm_embeds_i, input_ids1_i, input_ids2_i])
        self.size = len(self.data)
        print(f"Total filtered examples {self.size}")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        # Get sample from dataset
        mm_embeds, input_ids1, input_ids2 = self.data[i]
        return mm_embeds, input_ids1, input_ids2
            

class EEGInferenceDataset:

    # Constructor
    def __init__(self, args, captions=None):
        self.args = args
        self.captions = captions
        # Load EEG signals
        loaded = torch.load(args.eeg_dataset,weights_only=False)
        if args.subject != 0:
            self.data = [
                loaded["dataset"][i]
                for i in range(len(loaded["dataset"]))
                if loaded["dataset"][i]["subject"] == args.subject
            ]
        else:
            self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.channels = loaded["channels"]
        self.times = loaded["times"]

        # Compute size
        self.size = len(self.data)
        self.image_dir = args.image_dir

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG - use modern preprocessing like other datasets
        eeg = self.data[i]["eeg"].float()
        eeg = eeg.view(1, len(self.channels), len(self.times))
        
        label = self.data[i]["label"]
        label_string = self.labels[label]
        image_name = self.images[self.data[i]["image"]]
        image_id = self.data[i]["image"]
        
        # Use modern image path structure
        if label < 1654:
            image_path = os.path.join(
                self.image_dir, "training_images", label_string, image_name
            )
        else:
            image_path = os.path.join(
                self.image_dir, "test_images", label_string, image_name
            )
        
        # Load caption from precomputed list (required)
        if self.captions is None:
            raise RuntimeError(
                "Captions are required but not provided. "
                "Please provide precomputed captions indexed by image_id."
            )
        caption_raw = self.captions[image_id]

        return eeg, label_string, caption_raw, image_path, image_id


class SplitterInference:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # f.dataset = dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path,weights_only=False)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Compute size
        self.size = len(self.split_idx)
        print(f"Total examples in the split {split_name} {self.size}")

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label_string, expected_caption, image_path, image_id = self.dataset[
            self.split_idx[i]
        ]
        return eeg, label_string, expected_caption, image_path, image_id
