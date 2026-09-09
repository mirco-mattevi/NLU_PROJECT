class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        # batch size B, sequence length L, d_model
        B, L, d_model = x.size() 

        q = self.w_q(x) # (B, L, d_model)
        k = self.w_k(x) # (B, L, d_model)
        v = self.w_v(x) # (B, L, d_model)

        # reshape to (B, n_heads, L, h_dim)
        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        # dot-product between each head's q and k matrices
        # (B, n_heads, L, h_dim) * (B, n_heads, h_dim, L) -> (B, n_heads, L, L), i.e., similarities for each head
        similarity = q @ k.transpose(-2, -1) # transpose(-2,-1) transposes the k matrix for each head

        # normalize
        similarity = similarity * (1 / torch.sqrt(torch.tensor(self.h_dim)))

        # mask
        similarity = similarity.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(similarity, dim=-1)

        y = attn @ v  # (B, n_heads, L, L) * (B, n_heads, L, h_dim) -> (B, n_heads, L, h_dim)
        y = y.transpose(1, 2) # (B, L, n_heads, h_dim)
        # concatenate outputs of each head (contiguous is necessary to use view())
        y = y.contiguous().view(B, L, d_model) # (B, L, d_model)
        y = self.out_proj(y)

        return y

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        # applied over each token independently
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        # layer norm, attention, residual
        x = x + self.attn(self.ln1(x), mask)
        # layer norm, feedforward, residual
        x = x + self.ff(self.ln2(x))
        return x

# only difference for lab04 are out layers
class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size, 
        slots_size, 
        n_intents, 
        # GPT2 default hyperparameters
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        # learnable token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # learnable positional embeddings
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        #### These are different from the Language Modeling case of Part 1
        # slot outputs (sequence labeling): one slot label for each input token
        self.slot_out = nn.Linear(d_model, slots_size)
        # intent output (sequence classification): classyfy the intent of the input sequence
        self.intent_out = nn.Linear(d_model, n_intents)

        # triangular matrix: (sub)token i will only be able to attend (sub)tokens j, 0 <= j <= i
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        # mask is not a learnable parameter, 
        # but needs to be moved with everything else when doing .to(device) 
        self.register_buffer("mask", mask)
    
    def forward(self, idx, seq_lens):
        B, L = idx.shape # batch size, sequence length
        # positional embedding at most for position self.pos_emb_size
        # longer sequences cannot be processed (the model never learned positional embeddings for positions greater than self.pos_emb_size)
        assert L <= self.pos_emb_size

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)

        mask = self.mask[:, :, :L, :L]

        for block in self.blocks:
            x = block(x, mask)

        # We are also predicting a slot for CLS
        # but it will be ignored, as we have a pad token in the ground truth
        x = self.ln_f(x)
        slots = self.slot_out(x)

        # get intent from last token (CLS)
        tmp = []
        # for sequence i in the batch, 
        # get the vector associated with its CLS token 
        # (last of the sequence)
        for i in range(x.shape[0]):
            tmp.append(x[i, seq_lens[i]-1]) # -1, we count from 0
        cls_tokens = torch.stack(tmp)
        # get logits for intents
        intent = self.intent_out(cls_tokens)

        return slots, intent # ((B, L, slot_size), (B, n_intents)

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []

    for i, batch in enumerate(data):
        optimizer.zero_grad() # Zeroing the gradient

        slots, intent = model(batch['utterances'], batch['slots_len'])
        slots = slots.permute(0,2,1) # We need this for computing the loss

        loss_intent = criterion_intents(intent, batch['intents']) # intent and label for intents
        loss_slot = criterion_slots(slots, batch['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses (simplest way). 
                                       # Is there another way to do that? -> wheighted sum
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for batch in data:
            slots, intents = model(batch['utterances'], batch['slots_len'])
            slots = slots.permute(0,2,1) # We need this for computing the loss
            loss_intent = criterion_intents(intents, batch['intents'])
            loss_slot = criterion_slots(slots, batch['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())

            # Intent inference
            # Get the most probable class
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in batch['intents'].tolist()] # get the ground truth of the intent with more probability in the batch
            # lists with all the predictions for the intents
            ref_intents.extend(gt_intents) # 
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots): 
                length = batch['slots_len'].tolist()[id_seq] - 1 # -1, we ignore the CLS

                utt_ids = batch['utterances'][id_seq][:length].tolist()
                gt_ids = batch['y_slots'][id_seq][:length].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids]
                utterance = [lang.id2word[elem] for elem in utt_ids]

                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def init_model():
    lr = 1 # This is definitely not good for AdamW

    # in this case we need also these things
    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot) # pad and cls have the same id
    n_intents = len(lang.intent2id)

    # Experiment also with a smaller or bigger model 
    model = GPT2(
        vocab_len,
        slots_len,
        n_intents,
        pos_emb_size=1024,
        d_model=20,
        n_heads=1,
        num_layers=1,
        ff_dim=20,
    ).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN) # don't predict pad tokens
    criterion_intents = nn.CrossEntropyLoss() # No pad tokens, all sequences have a single label for intent

def train_NN():
    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    pbar = tqdm(range(n_epochs))
    for i in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model)
        if i % 1 == 0: # We check the performance every 1 epochs
            sampled_epochs.append(i)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, model, lang)

            pbar.set_description(f"Slot F1: {results_dev['total']['f']:.2f}; Intent Acc: {intent_res['abbreviation']['f1-score']:.2f}")
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break 

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                            criterion_intents, model, lang)    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

def save_model():
    PATH = os.path.join("bin", model_name)
    saving_object = {"epoch": x, 
                     "model": model.state_dict(), 
                     "optimizer": optimizer.state_dict(), 
                     "w2id": w2id, 
                     "slot2id": slot2id, 
                     "intent2id": intent2id}
    torch.save(saving_object, PATH)

def plot_train_and_valid_losses():
    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()
    plt.show()

def multiple_runs():
    lr = 1 # learning rate

    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot) # pad and cls have the same id
    n_intents = len(lang.intent2id)

    n_epochs = 200
    runs = 5

    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):
        model = GPT2(
            vocab_len,
            slots_len,
            n_intents,
            pos_emb_size=1024,
            d_model=20,
            n_heads=1,
            num_layers=1,
            ff_dim=20,
        ).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0

        pbar = tqdm(range(n_epochs))
        for x in pbar:
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)

                pbar.set_description(f"Slot F1: {results_dev['total']['f']:.2f}; Intent Acc: {intent_res['abbreviation']['f1-score']:.2f}")
                losses_dev.append(np.asarray(loss_dev).mean())

                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

def test_huggingface():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # not good parameters for alpha and rank
    model = AutoModel.from_pretrained("openai-community/gpt2")

    inputs = tokenizer(["I saw a man with a telescope", "StarLord was here",  "I didn't"], return_tensors="pt", padding=True)
    pprint(inputs)

    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)

    print(inputs["input_ids"][0])
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][1]))

def test_huggingface_bert():
    # BERT model script from: huggingface.co
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
    model = AutoModel.from_pretrained("bert-base-uncased") # Download the model

    inputs = tokenizer(["I saw a man with a telescope", "StarLord was here",  "I didn't"], return_tensors="pt", padding=True)
    pprint(inputs)

    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)


    print(inputs["input_ids"][0])
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][1]))