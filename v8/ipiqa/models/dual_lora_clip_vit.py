"""DualLoRACLIPViT -- 同一 frozen CLIP ViT 上维护 LoRA-Q / LoRA-A 两套适配器。

调整/改进7.md (V11): 拆掉单一 Visual LoRA，改为按 branch 选择不同 LoRA 参数，
切断 Quality / Alignment 在 backbone adaptation 层的梯度竞争。

实现：手动双 LoRA（不用 PEFT 多 adapter —— 其 set_adapter 会 toggle requires_grad，
导致前一 branch 的 LoRA 在反向被跳过）。这里把每层 q_proj/k_proj 替换为
DualLoRALinear：base 权重冻结，内部维护 lora_q / lora_a 两套独立参数，
两者恒为可训练；forward 时按 branch 选择使用哪一套。隔离完全由计算图保证：
quality 前向只经过 lora_q，alignment 前向只经过 lora_a。

vision 前向会被调用两次（quality + alignment），这是隔离的代价。
"""

import torch
import torch.nn as nn

from transformers import CLIPModel, CLIPTokenizer


class DualLoRALinear(nn.Module):
    """单个线性层上挂两套 LoRA（quality / alignment），branch 决定用哪套。"""

    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if base.bias is not None:
            self.base.bias.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.scaling = alpha / r
        self.branch = "quality"

        # 两套独立 LoRA 参数（始终可训练；隔离由计算图保证）
        self.lora_A_q = nn.Parameter(torch.empty(in_f, r))
        self.lora_B_q = nn.Parameter(torch.zeros(r, out_f))
        self.lora_A_a = nn.Parameter(torch.empty(in_f, r))
        self.lora_B_a = nn.Parameter(torch.zeros(r, out_f))
        nn.init.kaiming_uniform_(self.lora_A_q, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.lora_A_a, a=5 ** 0.5)

    def forward(self, x):
        y = self.base(x)
        if self.branch == "quality":
            y = y + (x @ self.lora_A_q @ self.lora_B_q) * self.scaling
        else:
            y = y + (x @ self.lora_A_a @ self.lora_B_a) * self.scaling
        return y


class DualLoRACLIPViT(nn.Module):
    def __init__(
            self,
            model_name="ckpt/clip-vit-base-patch16",
            context_length=77,
            lora_r=4,
            lora_alpha=8,
            freeze_visual=True,
            freeze_text=True,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.visual_width = self.clip.config.vision_config.hidden_size   # 768
        self.text_width = self.clip.config.text_config.hidden_size       # 512
        self.projection_dim = self.clip.config.projection_dim            # 512
        self.context_length = context_length
        self.freeze_visual = freeze_visual
        self.freeze_text = freeze_text

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = context_length

        # ---- 替换 q_proj/k_proj 为 DualLoRALinear，base 冻结 ----
        self.lora_wrappers = []
        for layer in self.clip.vision_model.encoder.layers:
            for pname in ["q_proj", "k_proj"]:
                orig = getattr(layer.self_attn, pname)
                wrapper = DualLoRALinear(orig, r=lora_r, alpha=lora_alpha)
                setattr(layer.self_attn, pname, wrapper)
                self.lora_wrappers.append(wrapper)

        # ---- freeze：vision base（除 LoRA 外全冻结）+ text + projection ----
        if freeze_visual:
            for p in self.clip.vision_model.parameters():
                p.requires_grad = False
            for w in self.lora_wrappers:
                for p in w.parameters():
                    if p is not w.base.weight and p is not w.base.bias:
                        p.requires_grad = True
        if freeze_text:
            for p in self.clip.text_model.parameters():
                p.requires_grad = False
        for p in self.clip.visual_projection.parameters():
            p.requires_grad = False
        for p in self.clip.text_projection.parameters():
            p.requires_grad = False

        self._active_branch = "quality"

    def tokenize(self, text):
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

    def encode_text(self, input_ids, attention_mask):
        text_out = self.clip.text_model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        global_t = self.clip.text_projection(text_out.pooler_output)
        return {"global_t": global_t}

    def encode_image(self, pixel_values, lora_branch="quality"):
        self._active_branch = lora_branch
        for w in self.lora_wrappers:
            w.branch = lora_branch
        out = self.clip.vision_model(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        global_v = self.clip.visual_projection(out.pooler_output)
        return {
            "global_v": global_v,                       # [B, 512]
            "vision_hidden": out.hidden_states,         # list [B,197,768]
        }

    def lora_trainable_count(self, branch=None):
        n = 0
        for w in self.lora_wrappers:
            if branch in (None, "quality"):
                n += w.lora_A_q.numel() + w.lora_B_q.numel()
            if branch in (None, "alignment"):
                n += w.lora_A_a.numel() + w.lora_B_a.numel()
        return n
