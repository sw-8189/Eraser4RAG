import os
import torch
from relik import Relik
from relik.inference.data.objects import RelikOutput
from relik.inference.annotator import Relik
from torch.utils.data import Dataset
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.optim import Adam
import argparse
import json
from datasets import load_metric
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import nltk
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Stage2RLDataset(Dataset):
    def __init__(self, args, tokenizer, filename_lis):
        self.examples = []
        random.seed(42)
        all_data = []
        for filename in filename_lis:
            with open(filename, encoding="utf-8") as f:
                data = f.readlines()
                n = len(data)
                n = int(args.use_data_percent * n)
                # randomly sample n samples for debugging
                if n < len(data):
                    random.seed(42)
                    data = random.sample(data, n)
            all_data.extend(data)
        sys_prompt = '''Rewrite the given text to relace only the private tail entities with the corresponding relations from the provided private triples 
                                while preserving all head entities, closely mirroring the original text, ensuring the text retains relevant public information and any details not mentioned in the triples.'''
        for record in all_data:
            record = json.loads(record)
            if args.use_prefix:
                public = "\n\npublic triples: "
                private = "\n\nprivate triples: "
                text = "\n\ntext: "
            else:
                public = ""
                private = ""
                text = ""
            public_trps = record["public"]
            tik = 0
            for trp in public_trps:  
                public = public + "<csubj>{}<crel>{}<cobj>{}<ce>".format(trp[0], trp[1], trp[2])

            private_trps = record["privacy"]
            for trp in private_trps: 
                private = private + "<rsubj>{}<rrel>{}<robj>{}<re>".format(trp[0], trp[1], trp[2])

            # private_trps = process_private_trps(private_trps)
            # public_trps = process_private_trps(public_trps)
            new_ctx = False
            ctxs = []
            for ctx in record["ctxs"]:
                if len(ctx["public"])>4 and len(ctx["private"])>1:
                    new_ctx = True
                    ctxs.append(ctx)
                    # break
            if not new_ctx:
                continue
            for ctx in ctxs:
                public_trps = set(ctx["public"])
                private_trps = set(ctx["private"])
                context = text + ctx["text"]
                input_text = sys_prompt + context + private + public + "**Anonymized Text**:"
                input_encoding = tokenizer.encode_plus(
                    input_text,
                    max_length=args.max_concat_length,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                new_example = {"input_ids": input_encoding['input_ids'].flatten(),
                               "attention_mask": input_encoding['attention_mask'].flatten(),
                               "private": private_trps,
                               "public": public_trps,
                               "text": ctx["text"]}
                self.examples.append(new_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def process_private_trps(private_trps):
    input_privacy = []
    for trp in private_trps: 
        input_privacy.append("<subj>{}<rel>{}<obj>{}<e>".format(trp[0], trp[1], trp[2]))
    return input_privacy


def replace_pub_tags(text):
    # replace <subj> with <csubj>
    text = text.replace('<subj>', '<csubj>')
    # replace <rel> with <crel>
    text = text.replace('<rel>', '<crel>')
    text = text.replace('<obj>', '<cobj>')
    text = text.replace('<e>', '<ce>')
    return text


def replace_pri_tags(text):
    # replace <subj> with <csubj>
    text = text.replace('<subj>', '<rsubj>')
    # replace <rel> with <crel>
    text = text.replace('<rel>', '<rrel>')
    text = text.replace('<obj>', '<robj>')
    text = text.replace('<e>', '<re>')
    return text


def compute_metrics(pred, label, args):
    # # Rouge expects a newline after each sentence
    decoded_pred = "\n".join(nltk.sent_tokenize(pred.strip()))

    decoded_label = "\n".join(nltk.sent_tokenize(label.strip()))
    result = metric.compute(predictions=[decoded_pred], references=[decoded_label],
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure for key, value in result.items()}['rougeL']
    result = result if result <= args.b else args.b

    return result / args.b


def extract_trps(bt_text):
    def extract(each_relik):
        triplets = each_relik.triplets
        true_triplets = []
        for triplet in triplets:
            t = "<subj>{}<rel>{}<obj>{}<e>".format(triplet.subject.text, triplet.label, triplet.object.text)
            if (t not in true_triplets) and (triplet.subject.text != triplet.object.text):
                true_triplets.append(t)
        return true_triplets
        
    logger.info(f"text = {bt_text[0]}")
    with torch.no_grad():
        relik_lis = relik(bt_text)
        bt_true_triplets = [extract(each_relik) for each_relik in relik_lis]
       
    return bt_true_triplets


def calculate_hit_reward(bt_true_triplets, bt_private, bt_public, p):
    bt_true_set = [set(true_triplets) for true_triplets in bt_true_triplets]
    bt_num_hit_private = [len(true_set & private) for (true_set,private) in zip(bt_true_set,bt_private)]
    bt_num_hit_public = [len(true_set & public) for (true_set,public) in zip(bt_true_set,bt_public)]
   
    bt_num_pub = [len(public) for public in bt_public]
    bt_num_pri = [len(private) for private in bt_private]
   
    bt_hit_ratio_private = [num_hit_private / num_pri for (num_hit_private, num_pri) in zip(bt_num_hit_private, bt_num_pri)]
    bt_hit_ratio_public = [num_hit_public / num_pub for (num_hit_public, num_pub) in zip(bt_num_hit_public, bt_num_pub)]
  
    logger.info(f"hit_ratio_public = {bt_hit_ratio_public[0]}, hit_ratio_private = {bt_hit_ratio_private[0]}")
    reward = [hit_ratio_public*torch.exp(-p*torch.tensor(hit_ratio_private)) for (hit_ratio_public, hit_ratio_private) in zip(bt_hit_ratio_public, bt_hit_ratio_private)]
    return reward


def all_zero(lis):
    sum_all = 0
    for i in lis:
        if i >= 1.0:
            sum_all += 1
    if sum_all>13:
        return True
    else:
        return False


def no_positive(lis):
    for i in lis:
        if i > 0.0:
            return False
    return True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir_1", type=str, default="./dataset/sample_privacy/popqa_10_25_trp.jsonl")
    parser.add_argument("--data_dir_2", type=str, default="./dataset/sample_privacy/triviaqa_10_25_trp.jsonl")
    parser.add_argument("--data_dir_3", type=str, default="./dataset/sample_privacy/NQ-open_10_25_trp.jsonl")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="./output_checkpoint/RL/")
    parser.add_argument("--logging_dir", type=str, default="output_checkpoint/RL/")
    parser.add_argument("--use_data_percent", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_prefix", type=bool, default=True)
    parser.add_argument("--max_passage_length", type=int, default=128)
    parser.add_argument("--max_concat_length", type=int, default=1300)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--show_freq", type=int, default=50)
    parser.add_argument("--p_reward", type=int, default=20)
    parser.add_argument("--static", type=bool, default=False)
    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    args.device = device

    return args


if __name__ == "__main__":
    args = get_args()
    relik = Relik.from_pretrained(
        "./relik-relation-extraction",
        device="cuda",
        use_nme=True
    )

    print("Loading Flan-T5 model...")
    tokenizer = AutoTokenizer.from_pretrained("./flan-t5-large")
    policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_dir).to(
        args.device)
    logger.info(args.model_dir)
    ref_policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(args.model_dir).to(
        args.device)
    special_tokens = ["<csubj>", "<crel>", "<cobj>", "<ce>", "<rsubj>", "<rrel>", "<robj>", "<re>"]
    tokenizer.add_tokens(special_tokens)
    tokenizer.save_pretrained("./flan-t5-large")

    print("Loading data...")
    train_dataset = Stage2RLDataset(args, tokenizer, [args.data_dir_1, args.data_dir_2, args.data_dir_3])
    logger.info(f"len_dataset={len(train_dataset)}\n")
    ppo_config = PPOConfig(
        seed=0,
        log_with="tensorboard",
        project_kwargs={"logging_dir": args.logging_dir},
        # adap_kl_ctrl=True,
        init_kl_coef=0.01,
        target=6,
        batch_size=16,
        mini_batch_size=8,
        backward_batch_size=8,
        ppo_epochs=4,
        learning_rate=1e-5,
        kl_penalty='kl',
        steps=200000,
        vf_coef=5,
        gradient_accumulation_steps=2,
        is_encoder_decoder=True
        # gamma=0.99,
        # whiten_rewards=True
    )
    metric = None  # load_metric("rouge")
    optimizer = Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=ppo_config.learning_rate,
    )
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = ppo_config.learning_rate
    scheduler= StepLR(optimizer, step_size=20, gamma=0.99, last_epoch=1) #LRScheduler(optimizer)#, last_epoch=1)
    ppo_trainer = PPOTrainer(ppo_config, policy, ref_policy, tokenizer, dataset=train_dataset,
                             data_collator=collator, optimizer=optimizer, lr_scheduler=scheduler)
    generation_kwargs = {
        "max_length": args.max_passage_length,
        "min_length": args.max_passage_length // 2,  # don't ignore the EOS token (see above)
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
        "do_sample": True,  # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id,
    }
    num_rounds = ppo_config.total_ppo_epochs // len(ppo_trainer.dataloader) + 1
    steps = 1

    p_reward = args.p_reward
    for rounds in range(num_rounds):
        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            prompt_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(
                prompt_tensors, return_prompt=False, generate_ref_response=False, **generation_kwargs
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
     
            if (epoch + rounds * len(ppo_trainer.dataloader)) % args.show_freq == 0:
                logger.info(f"res: {batch['response'][0]}")

            all_trps = extract_trps(batch["response"]) # + batch["ref_response"])
            bt_true_triplets = all_trps[:ppo_config.batch_size]
            if steps%350==0 and p_reward<=40 and args.static==False:
                p_reward+=5
            logger.info(f"p={p_reward}")
            batch_rewards = calculate_hit_reward(bt_true_triplets, batch["private"], batch["public"],p_reward)

            start = time.time()
            stats = ppo_trainer.step(prompt_tensors, response_tensors, batch_rewards)
            logger.info(f"steping time = {time.time()-start}")
            steps += 1
            ppo_trainer.log_stats(stats, batch, batch_rewards)
            if args.save_freq and epoch and (epoch + rounds * len(ppo_trainer.dataloader)) % args.save_freq == 0:
                ppo_trainer.save_pretrained(args.output_dir + f"step_{epoch + rounds * len(ppo_trainer.dataloader)}")
    ppo_trainer.save_pretrained(args.output_dir + f"step_final")


