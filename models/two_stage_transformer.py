#!/usr/bin/env python3
"""Two-stage transformer modules (stepwise implementation).

Step 1 implemented here: attribute tokenization, small attribute-level
Transformer, attention pooling and projection to element embedding.

This module purposely keeps Stage 2 (element-level Transformer) as a
separate next step. The AttributeStage takes per-attribute tensors and
returns a per-element embedding tensor of shape (B, S, D_elem).

Attribute mapping and shapes
----------------------------
The preprocessing script `prepare_poster_inputs.py` produces a concatenated
per-element feature vector X with shape (B, S, F) and an accompanying
`poster_inputs_schema.json` that defines offsets for fields. This module
expects those fields and provides a helper `slice_X_by_schema` to extract
the attributes below.

Inputs (per attribute):
 - image:  (B, S, img_dim)    e.g. img_dim=512 (CLIP image features)
 - text:   (B, S, text_dim)   e.g. text_dim=512 (CLIP text features)
 - pos:    (B, S, 2)          (left, top) normalized
 - size:   (B, S, 2)          (width, height) normalized
 - angle:  (B, S, 1)          normalized by 360
 - opacity:(B, S, 1)          clipped to [0,1]
 - font_idx:(B, S)            integer font ids (dense 0..K-1)

Tokenization / Stage 1:
 - Each attribute is projected to an attribute token of dim `d_attr` via
     dedicated Linear layers (image/text/pos/size/angle/opacity) and a
     font Embedding + projection. Tokens are stacked in a fixed order
     (img, text, pos, size, angle, opacity, font) producing shape
     (B, S, N_attr=7, d_attr).
 - A small attribute-level Transformer runs per-element on the N_attr
     tokens (flattening B*S -> batch dimension) and outputs tokens_out
     (B*S, N_attr, d_attr).
 - Attention pooling converts tokens_out -> pooled vector (B*S, d_attr)
 - pooled is projected to the element embedding dim D_elem and reshaped
     to (B, S, D_elem) which is the output of AttributeStage.

Notes:
 - The helper `find_poster_inputs` prefers per-split files named
     `poster_inputs_<split>_X.npy` and falls back to legacy names.
 - Use `slice_X_by_schema(X, schema)` to obtain attribute tensors before
     passing them to `AttributeStage`.
"""
from typing import Optional

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeTokenizer(nn.Module):
    """Project per-attribute inputs into a shared attribute embedding space.

    Expects callers to pass attribute tensors separately. This keeps the
    tokenizer simple and explicit.
    """

    def __init__(
        self,
        img_dim: int = 512,
        txt_dim: int = 512,
        pos_dim: int = 2,
        size_dim: int = 2,
        angle_dim: int = 1,
        opacity_dim: int = 1,
        font_emb_dim: int = 32,
        d_attr: int = 256,
        num_fonts: int = 1,
    ) -> None:
        super().__init__()
        self.img_proj = nn.Linear(img_dim, d_attr)
        self.txt_proj = nn.Linear(txt_dim, d_attr)
        self.pos_proj = nn.Linear(pos_dim, d_attr)
        self.size_proj = nn.Linear(size_dim, d_attr)
        self.angle_proj = nn.Linear(angle_dim, d_attr)
        self.opacity_proj = nn.Linear(opacity_dim, d_attr)
        # Per-attribute linear projectors: using separate projection matrices
        # makes tokens for different attribute types (image/text/pos/...)\
        # distinguishable to the attribute-level transformer.

        # font lookup + projection
        self.font_emb = nn.Embedding(num_fonts, font_emb_dim)
        self.font_proj = nn.Linear(font_emb_dim, d_attr)
        # Font is handled via an index embedding followed by a projection
        # so font tokens are brought into the same d_attr space as others.

        # small learned attribute type embeddings (to add to tokens)
        self.register_buffer("attr_type_ids", torch.arange(7))
        self.attr_type_emb = nn.Embedding(7, d_attr)
        # Explicit attribute-type embeddings are added to each token
        # (similar to BERT token-type embeddings) to provide an
        # additional, explicit signal about which attribute a token
        # represents.

        # learned per-attribute mask tokens (one token per attribute type)
        # used to replace attribute tokens for intentionally masked slots
        self.mask_token = nn.Parameter(torch.randn(7, d_attr) * 0.02)

        self.d_attr = d_attr

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        pos: torch.Tensor,
        size: torch.Tensor,
        angle: torch.Tensor,
        opacity: torch.Tensor,
        font_idx: torch.Tensor,
        slot_attr_mask: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return stacked attribute tokens of shape (B, S, N_attr, d_attr).

        Inputs shapes:
          img: (B, S, img_dim)
          txt: (B, S, txt_dim)
          pos: (B, S, pos_dim)
          size: (B, S, size_dim)
          angle: (B, S, 1)
          opacity: (B, S, 1)
          font_idx: (B, S) long
        """
        B, S, _ = img.shape
        # project
        img_tok = self.img_proj(img)
        txt_tok = self.txt_proj(txt)
        pos_tok = self.pos_proj(pos)
        size_tok = self.size_proj(size)
        angle_tok = self.angle_proj(angle)
        opacity_tok = self.opacity_proj(opacity)

        # font embedding (clamp indices safely)
        font_idx_safe = font_idx.clamp(min=0, max=self.font_emb.num_embeddings - 1)
        f_emb = self.font_emb(font_idx_safe)
        font_tok = self.font_proj(f_emb)

        # stack tokens in a fixed order
        tokens = torch.stack(
            [img_tok, txt_tok, pos_tok, size_tok, angle_tok, opacity_tok, font_tok], dim=2
        )
        # add attr-type embeddings
        # attr_type_emb has shape (7, d_attr) -> broadcast to (B,S,7,d_attr)
        atype = self.attr_type_emb(self.attr_type_ids.to(tokens.device))
        tokens = tokens + atype.view(1, 1, -1, self.d_attr)
        # Because tokens are stacked in a fixed order and we add an
        # attribute-type embedding, the attribute transformer can learn
        # index-specific behaviors (i.e. token 1 = text, token 2 = pos).

        # Determine per-attribute mask of shape (B, S, N_attr)
        # Priority: slot_attr_mask (explicit per-attribute) else slot_mask (broadcasted)
        if slot_attr_mask is not None:
            mam = slot_attr_mask
            if mam.dtype != torch.bool:
                mam = mam.to(torch.bool)
            if mam.dim() != 3:
                raise ValueError('slot_attr_mask must have shape (B, S, N_attr)')
        elif slot_mask is not None:
            sm = slot_mask
            if sm.dtype != torch.bool:
                sm = sm.to(torch.bool)
            mam = sm.unsqueeze(-1).expand(-1, -1, self.attr_type_ids.numel())
        else:
            mam = None

        if mam is not None:
            # broadcast to token shape (B, S, N_attr, 1)
            mask_b = mam.view(mam.size(0), mam.size(1), mam.size(2), 1)
            mt = self.mask_token.view(1, 1, -1, self.d_attr).to(tokens.device)
            mt = mt + atype.view(1, 1, -1, self.d_attr)
            # Replace masked attribute tokens with a learned, attribute-
            # specific mask token (plus the attr-type embedding) so the
            # transformer sees a consistent masked signal.
            tokens = torch.where(mask_b, mt, tokens)
        return tokens


class AttributeTransformer(nn.Module):
    """Small TransformerEncoder used to model intra-element attribute interactions.

    The transformer is shared across elements. Input shape (B*S, N_attr, d_attr).
    """

    def __init__(self, d_attr: int = 256, n_heads: int = 4, num_layers: int = 2, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_attr, nhead=n_heads, dim_feedforward=d_attr * mlp_mult, dropout=dropout, activation="gelu"
        )
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B*S, N_attr, d_attr) -> returns same shape"""
        # Transformer expects (seq_len, batch, emb) style for nn.TransformerEncoder.
        t = tokens.transpose(0, 1)  # (N_attr, B*S, d_attr)
        out = self.net(t)
        out = out.transpose(0, 1)  # (B*S, N_attr, d_attr)
        return out


class AttentionPooler(nn.Module):
    """Attention pooling across attribute tokens to produce a single vector per element."""

    def __init__(self, d_attr: int = 256):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_attr))
        self.scale = 1.0 / (d_attr ** 0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B*S, N_attr, d_attr) -> pooled: (B*S, d_attr)"""
        # compute token-wise scores by dotting each token with the learned query
        # tokens: (B*S, N_attr, d_attr), query: (d_attr,)
        q = self.query
        attn_logits = torch.einsum("bnd,d->bn", tokens, q) * self.scale  # (B*S, N_attr)
        attn = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
        pooled = (tokens * attn).sum(dim=1)  # (B*S, d_attr)
        return pooled


class AttributeStage(nn.Module):
    """Wrap tokenizer + attribute transformer + pooling + projection to element embedding.

    Forward inputs are per-attribute tensors (see AttributeTokenizer.forward). Returns
    elem_emb of shape (B, S, D_elem).
    """

    def __init__(
        self,
        img_dim: int = 512,
        txt_dim: int = 512,
        pos_dim: int = 2,
        size_dim: int = 2,
        angle_dim: int = 1,
        opacity_dim: int = 1,
        num_fonts: int = 1,
        font_emb_dim: int = 32,
        d_attr: int = 256,
        D_elem: int = 512,
        **kw,
    ) -> None:
        super().__init__()
        self.tokenizer = AttributeTokenizer(
            img_dim=img_dim,
            txt_dim=txt_dim,
            pos_dim=pos_dim,
            size_dim=size_dim,
            angle_dim=angle_dim,
            opacity_dim=opacity_dim,
            font_emb_dim=font_emb_dim,
            d_attr=d_attr,
            num_fonts=num_fonts,
        )
        self.attr_transformer = AttributeTransformer(d_attr=d_attr, **kw)
        self.pooler = AttentionPooler(d_attr=d_attr)
        self.elem_proj = nn.Linear(d_attr, D_elem)
        self.D_elem = D_elem

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        pos: torch.Tensor,
        size: torch.Tensor,
        angle: torch.Tensor,
        opacity: torch.Tensor,
        font_idx: torch.Tensor,
        slot_attr_mask: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-element embeddings.

        Returns: elem_emb (B, S, D_elem)
        """
        B, S, _ = img.shape
        tokens = self.tokenizer(
            img, txt, pos, size, angle, opacity, font_idx, slot_attr_mask=slot_attr_mask, slot_mask=slot_mask
        )
        # merge batch and slot dims to run transformer: (B*S, N_attr, d_attr)
        BS = B * S
        tokens_flat = tokens.view(BS, tokens.size(2), tokens.size(3))
        tokens_out = self.attr_transformer(tokens_flat)
        pooled = self.pooler(tokens_out)  # (B*S, d_attr)
        elem = self.elem_proj(pooled).view(B, S, self.D_elem)
        # The pooled per-element vector now summarizes all attributes for
        # that slot. Projecting to D_elem gives the element-level token
        # consumed by the ElementStage.
        return elem


class ElementStage(nn.Module):
    """Element-level transformer that consumes per-element embeddings and role indices.

    Adds learned role (type) embeddings and optional slot positional embeddings,
    runs a TransformerEncoder across elements (slots) and returns contextualized
    element embeddings of shape (B, S, D_elem).

    Args:
        D_elem: element embedding dimension (must match AttributeStage output)
        num_roles: number of distinct role/type ids (from preprocessing schema)
        max_slots: maximum number of element slots (used for slot pos embedding)
        n_heads, num_layers, mlp_mult, dropout: transformer hyperparams
    """

    def __init__(
        self,
        D_elem: int = 512,
        num_roles: int = 1,
        max_slots: int = 64,
        n_heads: int = 8,
        num_layers: int = 4,
        mlp_mult: int = 4,
        dropout: float = 0.1,
        num_attributes: int = 0,
    ) -> None:
        super().__init__()
        self.D_elem = D_elem
        self.num_roles = max(1, int(num_roles))
        self.max_slots = int(max_slots)

        # role / type embeddings
        self.role_emb = nn.Embedding(self.num_roles, D_elem)
        # Role/type embeddings provide a learned per-role bias added to
        # each slot so the element transformer can condition on semantic
        # element type (e.g. title, body, footer).

        # slot positional embeddings
        self.slot_pos_emb = nn.Embedding(self.max_slots, D_elem)
        # Learned absolute slot-index embeddings (one per slot index).
        # These encode slot identity/order (slot 0 vs slot 1 ...) and are
        # added to each element token before the element transformer.

        # optional masked-attribute embedding (0 means no masked attribute)
        self.num_attributes = int(num_attributes)
        if self.num_attributes > 0:
            # index 0 -> no attribute masked; 1..num_attributes -> attribute id
            self.masked_attr_emb = nn.Embedding(self.num_attributes + 1, D_elem)
            # Masked-attribute embedding indicates which attribute (if
            # any) was masked for this slot; added so the transformer
            # knows what to reconstruct.

        # element-level transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_elem, nhead=n_heads, dim_feedforward=D_elem * mlp_mult, dropout=dropout, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, elem_emb: torch.Tensor, role_idx: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, masked_attr_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            elem_emb: (B, S, D_elem)
            role_idx: (B, S) integer role/type ids (optional)
            mask: (B, S) binary mask (1=valid, 0=pad) used to construct src_key_padding_mask

        Returns:
            contextual_elem: (B, S, D_elem)
        """
        B, S, D = elem_emb.shape
        assert D == self.D_elem, f"elem_emb dim ({D}) != D_elem ({self.D_elem})"

        device = elem_emb.device

        x = elem_emb

        # add masked-attribute embedding if provided (masked_attr_id: (B, S) long, 0=no attr)
        if masked_attr_id is not None:
            if self.num_attributes <= 0:
                raise RuntimeError('ElementStage was not constructed with num_attributes>0 but masked_attr_id was provided')
            aid = masked_attr_id.clamp(min=0, max=self.num_attributes).to(device).long()
            aemb = self.masked_attr_emb(aid)
            # aemb: (B, S, D_elem) where index 0 means "no mask".
            x = x + aemb

        # add role embeddings if provided
        if role_idx is not None:
            # clamp safe
            rid = role_idx.clamp(min=0, max=self.role_emb.num_embeddings - 1).long()
            r = self.role_emb(rid.to(device))
            # add per-slot role/type embedding
            x = x + r

        # add slot positional embeddings
        pos_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        pos_e = self.slot_pos_emb(pos_idx)
        # add per-slot positional embedding (slot index)
        x = x + pos_e

        # prepare transformer input: (S, B, D)
        t_in = x.transpose(0, 1)

        # src_key_padding_mask expects True in positions that should be masked (padding)
        src_key_padding_mask = None
        if mask is not None:
            # mask: (B, S) with 1=valid, 0=pad -> invert to True for pad
            src_key_padding_mask = (mask == 0)
            # ensure bool on correct device
            src_key_padding_mask = src_key_padding_mask.to(device=device)

        t_out = self.transformer(t_in, src_key_padding_mask=src_key_padding_mask)
        out = t_out.transpose(0, 1).contiguous()
        return out


__all__ = ["AttributeTokenizer", "AttributeTransformer", "AttentionPooler", "AttributeStage", "ElementStage"]


def slice_X_by_schema(X, schema, device: Optional[torch.device] = None):
    """Slice a concatenated feature tensor X using a schema produced by preprocessing.

    Args:
        X: np.ndarray or torch.Tensor with shape (B, S, F)
        schema: dict containing 'fields' list with entries {'name','dim','offset'}
        device: optional torch.device to move tensors to (defaults to CPU)

    Returns:
        Tuple of torch.Tensor: (img, text, pos, size, angle, opacity)
        All tensors are float32 and have shapes:
            img: (B, S, img_dim)
            text: (B, S, text_dim)
            pos: (B, S, pos_dim)
            size: (B, S, size_dim)
            angle: (B, S, 1)
            opacity: (B, S, 1)
    """
    if torch.is_tensor(X):
        arr = X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        arr = X
    else:
        raise TypeError('X must be a numpy array or torch tensor')

    fields = {f['name']: (f['offset'][0], f['offset'][1]) for f in schema['fields']}
    def _slice(name):
        if name not in fields:
            raise KeyError(f"Schema missing required field: {name}")
        a, b = fields[name]
        sub = arr[:, :, a:b].astype(np.float32)
        t = torch.from_numpy(sub)
        if device is not None:
            t = t.to(device)
        return t

    img = _slice('image')
    text = _slice('text')
    pos = _slice('pos')
    size = _slice('size')
    angle = _slice('angle')
    opacity = _slice('opacity')
    return img, text, pos, size, angle, opacity


def find_poster_inputs(base_dir: str, prefer_splits=('test', 'validation', 'train')):
    """Find poster input files in base_dir preferring per-split filenames.

    Tries, in order, files named:
      poster_inputs_<split>_X.npy, poster_inputs_<split>_mask.npy, poster_inputs_<split>_font_idx.npy, poster_inputs_<split>_schema.json
    for each split in prefer_splits. If none are found, falls back to the legacy
    poster_inputs_X.npy / poster_inputs_mask.npy / poster_inputs_font_idx.npy / poster_inputs_schema.json.

    Returns (prefix, x_path, mask_path, font_path, schema_path) where prefix is None for legacy filenames
    or the split name when per-split files were found. Raises FileNotFoundError if no matching set exists.
    """
    # try per-split names first
    for sp in prefer_splits:
        x = os.path.join(base_dir, f'poster_inputs_{sp}_X.npy')
        m = os.path.join(base_dir, f'poster_inputs_{sp}_mask.npy')
        f = os.path.join(base_dir, f'poster_inputs_{sp}_font_idx.npy')
        s = os.path.join(base_dir, f'poster_inputs_{sp}_schema.json')
        if os.path.exists(x) and os.path.exists(m) and os.path.exists(f) and os.path.exists(s):
            return (sp, x, m, f, s)

    # fallback to legacy
    x = os.path.join(base_dir, 'poster_inputs_X.npy')
    m = os.path.join(base_dir, 'poster_inputs_mask.npy')
    f = os.path.join(base_dir, 'poster_inputs_font_idx.npy')
    s = os.path.join(base_dir, 'poster_inputs_schema.json')
    if os.path.exists(x) and os.path.exists(m) and os.path.exists(f) and os.path.exists(s):
        return (None, x, m, f, s)

    raise FileNotFoundError(f'No poster_inputs files found in {base_dir} (tried per-split and legacy names)')

