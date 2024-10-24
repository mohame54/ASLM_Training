from torch import nn
import torch
from torch.nn import functional as F
from typing import Union, Optional, List, Tuple
from utils import load_pretrained, print_trainable_parameters
from peft import LoraConfig, get_peft_model
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


class ASLM(nn.Module):
    def __init__(self, model_id="meta-llama/Llama-3.2-1B",logs=True,num_adds=11744):
        super().__init__()
        model = load_pretrained(model_id,logs=logs)
        config = model.config
        self.model = model
        self.vocab_size = config.vocab_size
        self.use_lm_head = False
        self.use_embed_layer = False
        self.padding_idx = config.pad_token_id
        self.config = config
        self.resize_lm_head(num_adds)
        self.resize_embedding(num_adds)

        # Initialize weights and apply final processing
    @torch.no_grad()
    def resize_lm_head(self, num_adds=12000):
        self.register_parameter("new_lm_weights", nn.Parameter(torch.randn((num_adds, self.config.hidden_size))))
        self.register_buffer("old_lm_weights", self.model.lm_head.weight.data.clone())
        self.old_lm_weights.requires_grad_(False)
        nn.init.trunc_normal_(self.new_lm_weights, std=0.2)

    @torch.no_grad()
    def resize_embedding(self, num_adds=12000):
        self.register_parameter("new_embd_weights", nn.Parameter(torch.randn((num_adds, self.config.hidden_size))))
        self.register_buffer("old_embd_weights", self.model.model.embed_tokens.weight.data.clone())
        self.old_embd_weights.requires_grad_(False)
        nn.init.trunc_normal_(self.new_embd_weights, std=0.2)

    def lm_logits(self, inputs):
        if not self.use_lm_head:
          weights = torch.cat([self.old_lm_weights, self.new_lm_weights], dim=0)
          return F.linear(inputs, weights)
        else:
          return self.model.lm_head(inputs)

    def embed_token_ids(self, inputs):
        if not self.use_embed_layer:
          weights = torch.cat([self.old_embd_weights, self.new_embd_weights], dim=0)
          return F.embedding(inputs, weights, self.padding_idx, norm_type=2.0)
        else:
          return self.model.model.embed_tokens(inputs)

    def combine_embedding(self):
        weights = torch.cat([self.old_embd_weights, self.new_embd_weights.data], dim=0)
        layer = nn.Embedding(self.config.vocab_size + self.num_adds, self.config.hidden_size, self.config.pad_token_id)
        layer.weight.data = weights
        self.model.model.embed_tokens = layer

    def combine_lm_head(self):
        weights = torch.cat([self.old_lm_weights, self.new_lm_weights.data], dim=0)
        layer = nn.Linear(self.config.hidden_size, self.config.vocab_size + self.num_adds, bias=False)
        layer.weight.data = weights
        self.model.lm_head = layer


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = self.embed_token_ids(input_ids)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        logits = self.lm_logits(hidden_states[:, -num_logits_to_keep:, :])
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



def make_peft_model(
    model,
    logs=True,
    lm_embd_new_weights=True,
    **kwargs
):
    params = dict(
          r=128,
          lora_alpha=32,
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
          lora_dropout=0.2,
          bias="none",
      )
    if len(kwargs) == 5:
      params = kwargs
    else:
      params.update(
          kwargs
      )  
    config = LoraConfig(
      **params
    )
    lora_model = get_peft_model(model, config)
    lora_model.base_model.new_embd_weights.requires_grad_(lm_embd_new_weights)
    lora_model.base_model.new_lm_weights.requires_grad_(lm_embd_new_weights)
    if logs:
      print("Setting Up the lora model with parameters", params)
      print_trainable_parameters(lora_model)
    return  lora_model  
