# model and training for 1B

# check how many parameters are trainable
def param_stats(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"frozen params: {total - trainable:,}")

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, rank, alpha):
        super().__init__(config)
        # Add layers to implement LoRA

    # edit the forward method to implement LoRa
    # from transformers 4.38.0
    # https://github.com/huggingface/transformers/blob/v4.38.0/src/transformers/models/gpt2/modeling_gpt2.py
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

# custum attention blocks (where I have to implement LoRA)
class GPT2_LoRA(GPT2LMHeadModel):
    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)
        # add custom attention blocks layers
        for block in self.transformer.h:
            # substitute block.attn with a new instance of CustomGPT2Attention
            # keep the weights from block.attn and apply them to the new instance using .load_state_dict()
            pass
    
    def forward(self, *args, **kwargs):
        return super().forward(*args,**kwargs)

def load_tokenizer_and_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2") # same tokenizer and data as part A (comparable)
    # not good parameters for alpha and rank
    model = GPT2_LoRA.from_pretrained("openai-community/gpt2", alpha=1, rank=1)
    model.to(DEVICE)

def train_loop(data, optimizer, model):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad() # Zeroing the gradient
        output = model(input_ids, labels=input_ids)
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        output.loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

        if i % 100 == 0:
            pbar.set_postfix(loss=(sum(loss_array)/sum(number_of_tokens)).item())

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for input_ids, labels, n_tokens in tqdm(data, desc="Evaluating: ", unit="batch", total=len(data)):
            output = model(input_ids, labels=input_ids)
            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)
            
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

def prepare_model_and_optimizer():
    lr = 0.1 # not good for AdamW

    vocab_len = len(tokenizer)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2_LoRA.from_pretrained("openai-community/gpt2", alpha=1, rank=1) # not good values for LoRA
    model.to(DEVICE)

    # train only adapter layers (LorA layers, L)
    for param in model.parameters():
        # start by freezing all parameters
        #param.requires_grad = False
        pass
    for module in model.modules():
        # make only your layers trainable
        #if hasattr(module, "<your_layer_name>"):
        #    for param in module.<your_layer_name>.parameters():
        #        param.requires_grad = True
        pass

    # check: print trainable layers
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr
    )

    param_stats(model) # #total params, #trainable params

def train_model():
    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(n_epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, model)    
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(loss.item())
            ppl_dev, loss_dev = eval_loop(dev_loader, model)
            losses_dev.append(loss_dev.item())
            pbar.set_description("PPL: %f" % ppl_dev)
            if ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break 

    best_model.to(DEVICE)
    final_ppl,  _ = eval_loop(test_loader, best_model)    
    print('Test ppl: ', final_ppl)