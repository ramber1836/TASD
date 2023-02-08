import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2PreTrainedModel, GPT2Tokenizer
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class GPT2XL(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config.n_embd, config.n_embd // 2)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        semantic_table_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[bool] = None,
        fixed_length: Optional[int] = 512,
        origin_gpt:Optional[bool] = False,
        relat_pos:Optional[bool] = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        if relat_pos:
            position_ids = position_ids % fixed_length #添加为了防止文本长度超过1024

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        # hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        

        cur_hidden_states = hidden_states[:, :fixed_length]
        all_hidden_states = []
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)): 
            outputs = block(cur_hidden_states)
            all_hidden_states.append(cur_hidden_states)
            cur_hidden_states = outputs[0]
        all_hidden_states.append(cur_hidden_states)


        for t in range((hidden_states.shape[1] - 1) // fixed_length):
            cur_hidden_states = hidden_states[:, (t + 1)*fixed_length :(t + 2)*fixed_length]
            for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)): 
                concat_hidden_states = torch.cat((all_hidden_states[i], cur_hidden_states), 1)
                outputs = block(concat_hidden_states)

                if origin_gpt:
                    #下面这种写法就和普通gpt2一致了，并且记得修改上面的位置编码
                    all_hidden_states[i] = concat_hidden_states
                    cur_hidden_states = outputs[0][:, (t + 1)*fixed_length:]
                else:
                    all_hidden_states[i] = cur_hidden_states
                    cur_hidden_states = outputs[0][:, fixed_length:]
                    # cur_hidden_states = self.tanh(self.linear(outputs[0]))

            all_hidden_states[-1] = torch.cat((all_hidden_states[-1], cur_hidden_states), 1)

        last_hidden_state = self.ln_f(all_hidden_states[-1])
        lm_logits = nn.functional.linear(last_hidden_state, self.wte.weight)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            # past_key_values=transformer_outputs.past_key_values,
            # hidden_states=transformer_outputs.hidden_states,
            # attentions=transformer_outputs.attentions,
            # cross_attentions=transformer_outputs.cross_attentions,
        )
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        semantic_table_ids: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        num_beam_groups: Optional[int] = None,
        fixed_length: Optional[int] = 512,
        origin_gpt:Optional[bool] = False,
        relat_pos:Optional[bool] = True,
    ):

        device = self.device
        pred = self.forward(inputs, fixed_length=fixed_length, origin_gpt=origin_gpt, relat_pos=relat_pos)[0]
        next_token = torch.tensor([torch.argmax(pred[...,-1,:])]).unsqueeze(0).to(device)

        while inputs.shape[1] < max_length and int(next_token) != eos_token_id:
            inputs = torch.cat((inputs, next_token), 1)
            pred = self.forward(inputs, fixed_length=fixed_length, origin_gpt=origin_gpt, relat_pos=relat_pos).logits
            next_token = torch.tensor([torch.argmax(pred[...,-1,:])]).unsqueeze(0).to(device)
            
        
        return inputs

def generate(m, tokenizer, text, max_length):
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    pred = m(**encoded_input)[0]
    next_token = tokenizer.decode(torch.argmax(pred[...,-1,:]))

    while encoded_input['input_ids'].shape[1] < max_length and next_token != "<endoftext>":
        encoded_input = tokenizer(text, return_tensors='pt').to(device)
        pred = m(**encoded_input)[0]
        next_token = tokenizer.decode(torch.argmax(pred[...,-1,:]))
        text += next_token
    
    return text

if __name__=='__main__':
    device = "cuda:0"
    max_length = 20
    with torch.no_grad():
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        model_lm = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        model_xl = GPT2XL.from_pretrained('gpt2').to(device)
        text = open("src/temp.txt", "r").readline()

        inputs = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
        semantic_table = None
        inputs = {
                'inputs': inputs,
                'semantic_table_ids': semantic_table,
                'max_length': inputs.shape[1] + max_length,
                'do_sample': False,
                'num_beams': 1,
                "early_stopping": True,
                "pad_token_id": 50256,
                "eos_token_id": 50256,
                "num_return_sequences": 1,
            }
        outputs_xl_origin = model_xl.generate(**inputs, fixed_length=100, origin_gpt=True, relat_pos=False)
        outputs_xl_normal = model_xl.generate(**inputs, fixed_length=100, origin_gpt=False)
        outputs_lm = model_lm.generate(**inputs)
        pred_text_xl_origin = [tokenizer.decode(outputs_xl_origin[i].tolist()) for i in range(len(outputs_xl_origin))]
        pred_text_xl_normal = [tokenizer.decode(outputs_xl_normal[i].tolist()) for i in range(len(outputs_xl_normal))]
        pred_text_lm = [tokenizer.decode(outputs_lm[i].tolist()) for i in range(len(outputs_lm))]
        
        print(pred_text_xl_origin)
        print(pred_text_xl_normal)
        print(pred_text_lm)