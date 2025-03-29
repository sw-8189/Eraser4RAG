import os
from data_structure import Anonymize_Doc_popqa
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
import argparse
import torch
from torch.utils.data import random_split
import numpy as np
import nltk
from datasets import load_metric

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset/constructed_dataset/popqa_10_25_filtered_true.jsonl")
    parser.add_argument("--model_dir", type=str, default="./flan-t5-large")
    parser.add_argument("--output_dir", type=str, default="./output_checkpoint")
    parser.add_argument("--use_data_percent", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_prefix", type=bool, default=True)
    parser.add_argument("--max_passage_length", type=int, default=128)
    parser.add_argument("--max_public_length", type=int, default=742)
    parser.add_argument("--max_private_length", type=int, default=252)
    parser.add_argument("--max_concat_length", type=int, default=1300)
    args = parser.parse_args()

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # , args.local_rank)
    args.device = device

    return args


if __name__ == '__main__':
    args = get_args()

    print("Loading Flan-T5 model...")
    tokenizer = AutoTokenizer.from_pretrained("./flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(args.device)
    special_tokens = ["<csubj>", "<crel>", "<cobj>", "<ce>","<rsubj>", "<rrel>", "<robj>", "<re>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    # tokenizer.save_pretrained(args.model_dir)
    print("Loading dataset...")
    dataset = Anonymize_Doc_popqa(args, tokenizer, args.data_dir)
    train_size = int(len(dataset) * 0.85)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        # predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # logging & evaluation strategies
        logging_first_step=True,
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
        predict_with_generate=True,
        generation_max_length=128,
        # metric_for_best_model="rouge1",
        save_steps=1000
    )
    print("Finetuning...")
    # metric = load_metric("rouge")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors='pt')
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    trainer.train()
