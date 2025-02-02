import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(42)

config = LlamaConfig( 
    vocab_size=32， 
    hidden_size=512,  
    intermediate_size=2,
    num_attention_heads=4,
    num_key_value_heads=4  
)

ref_model = LlamaForCausalLM(config) 
ref_model.eval()  
model = LlamaForCausalLM(config)
print(model.lm_head) 

prompt_length = 6 
answer_length = 4
prompt_chosen = torch.tensor([[5, 8, 9, 10, 5, 3, 16, 29, 18, 17]], dtype=torch.int64)
prompt_rejected = torch.tensor([[5, 8, 9, 10, 5, 3, 26, 14, 31, 0]], dtype=torch.int64)
attention_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool)

x_chosen = {'input_ids': prompt_chosen, 'attention_mask': attention_mask}
x_rejected = {'input_ids': prompt_rejected, 'attention_mask': attention_mask}

output= ref_model(**x_chosen) 

def get_probs(logits, labels):  
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return per_token_logps

probs_chosen = get_probs(output.logits, prompt_chosen)

print('logits形状为:\n', output.logits.shape)
print('chosen的最后一个id号码的token为:\n', prompt_chosen[0,-1])
print('chosen的最后一个id号的logits为:\n', output.logits[0,-1,:])
print('chosen的最后最后一个id号的logprob为:\n', output.logits[0,-1,:].log_softmax(-1))
print('chosen的最后最后一个id号的token logprob为:\n',output.logits[0,-1,:].log_softmax(-1)[prompt_chosen[0,-1]])
print('-'*50)
print('chosen数据为:\n', prompt_chosen)
print('chosen中每个token的logprob为:\n', probs_chosen)

logits_chosen_ref = ref_model(**x_chosen).logits  
logits_rejected_ref = ref_model(**x_rejected).logits  
logits_chosen = model(**x_chosen).logits  
logits_rejected = model(**x_rejected).logits  

probs_chosen_ref = get_probs(logits_chosen_ref, prompt_chosen)  
probs_chosen = get_probs(logits_chosen, prompt_chosen)
probs_rejected_ref = get_probs(logits_rejected_ref, prompt_rejected)
probs_rejected = get_probs(logits_rejected, prompt_rejected)

beta = 0.1
pi_logratios = probs_chosen - probs_rejected差
ref_logratios = probs_chosen_ref - probs_rejected_ref
logits = pi_logratios - ref_logratios
# logits = (probs_chosen - probs_rejected) - (probs_chosen_ref - probs_rejected_ref)
losses = -F.logsigmoid(beta * logits) * labels
print(losses)
loss = losses.sum(-1) / attention_mask.sum
print(loss)
