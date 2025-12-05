"""
Modality: Text
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from ..registry import register_text_embedding

class Pooler:
  @staticmethod
  def average_pool(
    last_hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor
  ) -> torch.FloatTensor:
    """
    Average pooling w/o attention mask
    """
    
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

@register_text_embedding("transformers")
class Transformers(nn.Module):
  """
  引用 huggingface transformers 支援的 models
  """

  def __init__(
    self,
    model_name: str = 'intfloat/e5-small-v2',
    freeze: bool = True
  ):
    """
    Args:
      model_name: Model name on huggingface hub. 
      freeze: Parameter tuning?
    """

    super(E5SentEmbedding, self).__init__()
    
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)
    
    if freeze:
      for p in self.model.parameters():
          p.requires_grad = False
    
  def forward(
    self, 
    input_texts: torch.FloatTensor
  ) -> torch.FloatTensor:
    
    # Tokenize the input texts
    batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.model.device)

    outputs = self.model(**batch_dict)
    embeddings = Pooler.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    return embeddings

@register_text_embedding("Sentence-Transformer")
class SentTransformer(nn.Module):
  """
  引用 sentence-transformer 支援的 models
  """

  def __init__(
    self, 
    model_name: str,
    freeze: bool = True
  ):
    """
    Args:
      model_name: Model name on sentence-transformer. 
      freeze: Parameter tuning?
    """
    
    super(SentTransformer, self).__init__()
    
    self.model = SentenceTransformer(model_name)

    if freeze:
      for param in self.model[0].parameters():  # model[0] = Transformer
          param.requires_grad = False
       
      for param in self.model[1].parameters():  # Pooling
          param.requires_grad = False
      
      for param in self.model[2].parameters():  # Dense layer (if exists)
         param.requires_grad = False
        
  def forward(
    self,
    input_text: torch.FloatTensor
  ) -> torch.FloatTensor:

    embeddings = model.encode(sentences)
  
    return embeddings

@register_text_embedding("LLM-Embedding")
class LLMEmbedding(nn.Module):
  """
  引用 LLM 的 embedding layer
  適用於多模態合併字串: [Text, Time-series, ...]
  """

  def __init__(
    self, 
    model_name: str,
    freeze: bool = True,
    max_new_tokens: int = 128, 
    do_sample: bool = True, 
    temperature: float = 0.6, 
    top_p: float = 0.9
  ):
    """
    Args:
      model_name: Model name on sentence-transformer. 
      freeze: Parameter tuning?
    """
    
    super(SentTransformer, self).__init__()
    
    self.llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, **kwargs)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    self.max_new_tokens = max_new_tokens
    self.do_sample = do_sample
    self.temperature = temperature
    self.top_p = top_p
    
    if freeze:
      for param in self.llm.parameters():
        param.requires_grad = False
        
  def forward(
    self,
    input_text: torch.FloatTensor,
    other_modality_position: torch.FloatTensor,
    other_modality_features: torch.FloatTensor
  ) -> torch.FloatTensor:

    inputs = self.process_text(input_text)
    
    inputs_embeds, attention_mask = self.prepare_llm_input(
        input_ids=inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        other_modality_position=other_modality_position,
        other_modality_features=other_modality_features
    )

    outputs = self.llama.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        pad_token_id=self.tokenizer.eos_token_id,
        max_new_tokens=self.max_new_tokens,
        do_sample=self.do_sample,
        temperature=self.temperature,
        top_p=self.top_p,
    )
    
    return outputs

  def process_text(
    self,
    input_text: str
  ):
    """
    產生 chat template + tokenization
    """

    messages = [
      {"role": "system", "content": "You are a helpful text processor"},
      {"role": "user", "content": input_text}
    ]
  
    context = self.tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
    )

    inputs = self.tokenizer(
      context,
      return_tensors="pt"
    )

    return inputs
  
  def prepare_llm_input(
    self, 
    input_ids: torch.FloatTensor, 
    attention_mask: torch.FloatTensor,
    other_modality_position: torch.FloatTensor,
    other_modality_features: torch.FloatTensor
  ):
    """
    產生 LLM 所需 input_embeds, attention mask
    """
    
    input_ids = input_ids.to(self.llm.device)
    attention_mask = attention_mask.to(self.llm.device)

    inputs_embeds = self.llm.model.embed_tokens(input_ids) # [bs, seq_len, hidden_size]

    inputs_embeds = torch.cat([inputs_embeds[0, :other_modality_position], audio_features[0, :], inputs_embeds[0, other_modality_position:]], dim=0)
    attention_mask = torch.cat([attention_mask[0, :other_modality_position], torch.ones([other_modality_features], dtype=torch.long, device=self.llm.device), attention_mask[0, other_modality_position:]], dim=0)

    inputs_embeds = inputs_embeds.to(self.llama.dtype)
    attention_mask = attention_mask.to(self.llama.dtype)
    
    return inputs_embeds.unsqueeze(0), attention_mask.unsqueeze(0)
