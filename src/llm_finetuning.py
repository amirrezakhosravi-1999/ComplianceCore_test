#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for fine-tuning LLMs.
This includes LoRA setup and training script for relation extraction and compliance judgment.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMFineTuner:
    '''Class for fine-tuning LLMs with LoRA/PEFT'''
    
    def __init__(self,
                 base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 dataset_path: str = '../data/fine_tuning_datasets',
                 output_dir: str = '../models/fine_tuned_llm',
                 task_type: str = 'relation_extraction'):
        '''
        Initialize the LLMFineTuner.
        
        Args:
            base_model_name: Name or path of the base model to fine-tune
            dataset_path: Path to the fine-tuning datasets directory
            output_dir: Directory to save fine-tuned models
            task_type: Type of fine-tuning task ('relation_extraction' or 'compliance_judgment')
        '''
        self.base_model_name = base_model_name
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.task_type = task_type
        self.tokenizer = None
        self.model = None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'LLMFineTuner initialized with base_model={base_model_name}, '
                   f'task_type={task_type}, output_dir={output_dir}')

    def load_tokenizer(self):
        '''Load the tokenizer for the base model'''
        logger.info(f'Loading tokenizer for {self.base_model_name}')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        # Ensure the tokenizer has padding token, eos token, and bos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'
            
        logger.info('Tokenizer loaded successfully')
        return self.tokenizer

    def load_model_for_training(self):
        '''Load the model with BitsAndBytes quantization and prepare for LoRA training'''
        logger.info(f'Loading model {self.base_model_name} for training')
        
        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load the model with quantization config
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True
        )
        
        # Prepare the model for training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        )
        
        # Create PEFT model
        model = get_peft_model(model, lora_config)
        
        self.model = model
        logger.info('Model loaded and prepared for training successfully')
        return self.model

    def prepare_relation_extraction_dataset(self, dataset_path: str = None):
        '''Prepare dataset for relation extraction fine-tuning'''
        if dataset_path is None:
            dataset_path = self.dataset_path / 'relations.jsonl'
            
        logger.info(f'Preparing relation extraction dataset from {dataset_path}')
        
        # Load the dataset
        dataset = load_dataset('json', data_files=str(dataset_path))
        
        def format_relation_sample(sample):
            '''Format a single sample for relation extraction fine-tuning'''
            entities = sample['entities']
            relations = sample['relations']
            
            # Create a formatted instruction with input and output
            instruction = (
                'Extract relations between entities from the following text. '
                'Identify all entity pairs and their relationship types.'
            )
            
            input_text = f"Text: {sample['text']}\n\nEntities: "
            for entity in entities:
                input_text += f"{entity['span']} ({entity['type']}), "
            input_text = input_text.rstrip(', ')
            
            output = 'Relations:\n'
            for relation in relations:
                output += f"- {relation['head']} -> {relation['type']} -> {relation['tail']}\n"
                
            return {
                'instruction': instruction,
                'input': input_text,
                'output': output
            }
            
        # Apply formatting to the dataset
        formatted_dataset = dataset['train'].map(format_relation_sample)
        
        logger.info(f'Dataset prepared with {len(formatted_dataset)} samples')
        return formatted_dataset

    def prepare_compliance_judgment_dataset(self, dataset_path: str = None):
        '''Prepare dataset for compliance judgment fine-tuning'''
        if dataset_path is None:
            dataset_path = self.dataset_path / 'compliance_checks.jsonl'
            
        logger.info(f'Preparing compliance judgment dataset from {dataset_path}')
        
        # Load the dataset
        dataset = load_dataset('json', data_files=str(dataset_path))
        
        def format_compliance_sample(sample):
            '''Format a single sample for compliance judgment fine-tuning'''
            # Create a formatted instruction with input and output
            instruction = (
                'Determine whether a design specification complies with a regulatory requirement. '
                'Provide a detailed reasoning for your judgment.'
            )
            
            input_text = (
                f"Regulatory Requirement: {sample['regulatory_clause']}\n\n"
                f"Design Specification: {sample['design_segment']}"
            )
            
            output = (
                f"Compliance Status: {sample['compliance_status']}\n\n"
                f"Reasoning: {sample['reasoning']}"
            )
                
            return {
                'instruction': instruction,
                'input': input_text,
                'output': output
            }
            
        # Apply formatting to the dataset
        formatted_dataset = dataset['train'].map(format_compliance_sample)
        
        logger.info(f'Dataset prepared with {len(formatted_dataset)} samples')
        return formatted_dataset

    def train(self, epochs: int = 3, learning_rate: float = 2e-5, batch_size: int = 4):
        '''Train the model with the fine-tuning dataset'''
        logger.info(f'Starting training for {self.task_type} with {epochs} epochs')
        
        # Load tokenizer and model if not already loaded
        if self.tokenizer is None:
            self.load_tokenizer()
        if self.model is None:
            self.load_model_for_training()
            
        # Prepare dataset based on task type
        if self.task_type == 'relation_extraction':
            dataset = self.prepare_relation_extraction_dataset()
        elif self.task_type == 'compliance_judgment':
            dataset = self.prepare_compliance_judgment_dataset()
        else:
            raise ValueError(f'Unsupported task type: {self.task_type}')
            
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / self.task_type),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            optim='paged_adamw_8bit',
            save_steps=50,
            logging_steps=10,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type='cosine',
            report_to='tensorboard'
        )
        
        # Set up the SFT trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            peft_config=self.model.peft_config,
            dataset_text_field='input',
            tokenizer=self.tokenizer,
            max_seq_length=2048
        )
        
        # Train the model
        logger.info('Starting training process')
        trainer.train()
        
        # Save the trained model
        output_path = str(self.output_dir / f'{self.task_type}_adapter')
        trainer.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f'Training completed, model saved to {output_path}')
        return output_path

    def evaluate(self, test_data_path: str = None):
        '''Evaluate the fine-tuned model on test data'''
        # This is a simplified evaluation for the PoC
        # In a real-world scenario, you would implement a more comprehensive evaluation
        logger.info('Evaluation not fully implemented in PoC')
        return {'message': 'Evaluation functionality to be implemented'}

    def create_example_dataset(self, output_path: str = None, num_samples: int = 10):
        '''Create an example dataset for fine-tuning demonstration'''
        if output_path is None:
            if self.task_type == 'relation_extraction':
                output_path = self.dataset_path / 'relations.jsonl'
            else:
                output_path = self.dataset_path / 'compliance_checks.jsonl'
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f'Creating example {self.task_type} dataset with {num_samples} samples at {output_path}')
        
        examples = []
        
        if self.task_type == 'relation_extraction':
            # Create example relation extraction dataset
            for i in range(num_samples):
                example = {
                    "text": f"The reactor coolant system shall be designed to withstand design basis accidents without exceeding specified safety limits. This is requirement #{i+1}.",
                    "entities": [
                        {"span": "reactor coolant system", "type": "SYSTEM"},
                        {"span": "design basis accidents", "type": "EVENT"},
                        {"span": "safety limits", "type": "PARAMETER"}
                    ],
                    "relations": [
                        {"head": "reactor coolant system", "type": "designed_to_withstand", "tail": "design basis accidents"},
                        {"head": "design basis accidents", "type": "constrained_by", "tail": "safety limits"}
                    ]
                }
                examples.append(example)
        else:  # compliance_judgment
            # Create example compliance judgment dataset
            compliance_statuses = ["Compliant", "Partially Compliant", "Non-Compliant"]
            for i in range(num_samples):
                status_idx = i % 3
                example = {
                    "regulatory_clause": f"The pump's casing shall be capable of withstanding {450 + i*10} psi.",
                    "design_segment": f"The pump casing is designed for {400 + i*20} psi.",
                    "compliance_status": compliance_statuses[status_idx],
                    "reasoning": f"The design pressure of {400 + i*20} psi is {'greater than' if 400+i*20 > 450+i*10 else 'less than'} the required {450 + i*10} psi."
                }
                examples.append(example)
        
        # Write examples to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
        logger.info(f'Created example dataset at {output_path}')
        return output_path


def main():
    '''Main function to demonstrate the fine-tuning process'''
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune LLM for regulatory compliance')
    parser.add_argument('--task', type=str, default='relation_extraction',
                        choices=['relation_extraction', 'compliance_judgment'],
                        help='Type of fine-tuning task')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--create-dataset', action='store_true',
                        help='Create an example dataset')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples in example dataset')
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = LLMFineTuner(
        base_model_name=args.model,
        task_type=args.task
    )
    
    # Create example dataset if requested
    if args.create_dataset:
        fine_tuner.create_example_dataset(num_samples=args.samples)
        logger.info('Example dataset created. Exiting without training.')
        return
    
    # Train the model
    model_path = fine_tuner.train(epochs=args.epochs)
    
    logger.info(f'Fine-tuning completed, model saved at {model_path}')


if __name__ == '__main__':
    main()
