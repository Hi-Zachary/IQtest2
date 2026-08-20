"""CLIP ViT-B/16 backbone with optional visual LoRA (v8).

Backbone adaptation strategy (调整/改进1.md section 4-5):
  - Frozen CLIP ViT-B/16 via HuggingFace ``transformers.CLIPModel``.
  - ``output_hidden_states=True`` gives per-layer patch/CLS tokens
    (hidden_states[0..12], each [B, 197, 768]).
  - Visual LoRA: only ``q_proj``/``k_proj``, r=4, alpha=8, dropout=0.05.
  - Text encoder always frozen.

Output dict:
  {
    "global_v":      visual_projection(pooler_output)   [B, 512]
    "global_t":      text_projection(pooler_output)     [B, 512]
    "vision_hidden": list of 13 hidden states           [B, 197, 768]
    "text_tokens":   text last_hidden_state             [B, 77, 512]
    "text_mask":     attention_mask (bool)              [B, 77]
  }
"""

import torch
import torch.nn as nn

from transformers import CLIPModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model


class CLIPViTBackbone(nn.Module):
    def __init__(
            self,
            model_name="ckpt/clip-vit-base-patch16",
            context_length=77,
            use_lora=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            freeze_visual=True,
            freeze_text=True,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.visual_width = self.clip.config.vision_config.hidden_size   # 768
        self.text_width = self.clip.config.text_config.hidden_size       # 512
        self.projection_dim = self.clip.config.projection_dim            # 512
        self.context_length = context_length

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = context_length

        self.use_lora = use_lora
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text

        # ---------- Visual LoRA（只包 vision_model，避免误加到 text encoder） ----------
        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj"],
                bias="none",
            )
            self.clip.vision_model = get_peft_model(self.clip.vision_model, lora_cfg)

        # ---------- freeze ----------
        # LoRA 模式下 peft 已冻结 base（只留 lora_A/lora_B 可训练）；
        # 非 LoRA 模式显式冻结全部视觉参数。
        if not use_lora and freeze_visual:
            for p in self.clip.vision_model.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.clip.text_model.parameters():
                p.requires_grad = False
        # projection 头属于冻结 backbone 的一部分（与 NR_IQA_AGM 一致）
        for p in self.clip.visual_projection.parameters():
            p.requires_grad = False
        for p in self.clip.text_projection.parameters():
            p.requires_grad = False

    def tokenize(self, text):
        """text: str or list[str] -> (input_ids [B,77], attention_mask [B,77] int64)."""
        if isinstance(text, str):
            text = [text]
        tok = self.tokenizer(
            list(text),
            padding="max_length",
            max_length=self.context_length,
            truncation=True,
            return_tensors="pt",
        )
        return tok.input_ids, tok.attention_mask

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_out = self.clip.vision_model(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        text_out = self.clip.text_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        global_v = self.clip.visual_projection(vision_out.pooler_output)
        global_t = self.clip.text_projection(text_out.pooler_output)

        return {
            "global_v": global_v,                       # [B, 512]
            "global_t": global_t,                       # [B, 512]
            "vision_hidden": vision_out.hidden_states,  # list [B,197,768]
            "text_tokens": text_out.last_hidden_state,  # [B,77,512]
            "text_mask": attention_mask.bool(),         # [B,77]
        }

    def lora_trainable_count(self):
        if not self.use_lora:
            return 0
        n = 0
        for p in self.clip.vision_model.parameters():
            if p.requires_grad:
                n += p.numel()
        return n
