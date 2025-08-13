"""Example: Using OpenRubricRL with TRL for language model fine-tuning."""

import os
import json
from typing import List, Dict, Any

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer
)

from openrubricrl.integrations.trl import (
    create_trl_reward_function,
    create_ppo_trainer_with_rubric,
    RubricCallback
)
from openrubricrl.logging import OpenRubricLogger, TrainingLogger


def create_sample_dialogue_rubric():
    """Create a sample rubric for dialogue evaluation."""
    rubric_path = "dialogue_quality.json"
    
    if not os.path.exists(rubric_path):
        rubric_data = {
            "name": "dialogue_quality",
            "version": "1.0.0",
            "description": "Evaluate dialogue response quality",
            "domain": "dialogue",
            "scale": {"min": 0.0, "max": 10.0},
            "criteria": [
                {
                    "name": "relevance",
                    "description": "Is the response relevant to the user's query?",
                    "weight": 0.4,
                    "examples": {
                        "excellent": [
                            {
                                "input": "What's the weather like today?",
                                "output": "I don't have access to real-time weather data, but you can check your local weather forecast on weather.com or your weather app.",
                                "score": 9.0,
                                "explanation": "Directly addresses the question and provides helpful alternatives"
                            }
                        ],
                        "poor": [
                            {
                                "input": "What's the weather like today?",
                                "output": "I like pizza.",
                                "score": 1.0,
                                "explanation": "Completely irrelevant to the question"
                            }
                        ]
                    }
                },
                {
                    "name": "helpfulness",
                    "description": "Does the response provide useful information?",
                    "weight": 0.3
                },
                {
                    "name": "clarity",
                    "description": "Is the response clear and well-written?",
                    "weight": 0.3
                }
            ],
            "hybrid_metrics": [
                {
                    "name": "response_length",
                    "type": "custom",
                    "weight": 0.1,
                    "config": {"min_length": 10, "max_length": 200}
                }
            ]
        }
        
        with open(rubric_path, 'w') as f:
            json.dump(rubric_data, f, indent=2)
        
        print(f"Created sample rubric: {rubric_path}")
    
    return rubric_path


def prepare_training_data() -> List[Dict[str, str]]:
    """Prepare sample training data for dialogue fine-tuning."""
    return [
        {
            "query": "How do I cook pasta?",
            "response": "To cook pasta: 1) Boil water in a large pot, 2) Add salt, 3) Add pasta and cook according to package directions, 4) Drain and serve."
        },
        {
            "query": "What's the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "query": "How do I fix a leaky faucet?",
            "response": "For a leaky faucet, try: 1) Turn off water supply, 2) Check and replace worn washers or O-rings, 3) Tighten connections. If problem persists, call a plumber."
        },
        {
            "query": "Tell me a joke",
            "response": "Why don't scientists trust atoms? Because they make up everything!"
        },
        {
            "query": "What's 2+2?",
            "response": "2 + 2 = 4"
        }
    ]


def basic_trl_example():
    """Basic example using TRL reward function."""
    
    # Set up logging
    logger = OpenRubricLogger(log_file="trl_training.log")
    
    try:
        # Create rubric
        rubric_path = create_sample_dialogue_rubric()
        
        # Create reward function
        reward_function = create_trl_reward_function(
            rubric_path=rubric_path,
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Test the reward function
        logger.log_info("Testing reward function...")
        
        test_queries = [
            "What's the weather like?",
            "How do I learn Python?", 
            "Tell me about machine learning"
        ]
        
        test_responses = [
            "I don't have real-time weather data, but you can check weather.com for your local forecast.",
            "Start with online tutorials, practice coding daily, and work on small projects.",
            "Machine learning is a subset of AI that enables computers to learn patterns from data."
        ]
        
        for query, response in zip(test_queries, test_responses):
            reward = reward_function(query, response)
            detailed = reward_function.get_detailed_reward(query, response)
            
            logger.log_info(f"Query: {query}")
            logger.log_info(f"Response: {response}")
            logger.log_info(f"Reward: {reward:.3f}")
            logger.log_info(f"Explanation: {detailed.explanation}")
            logger.log_info("---")
        
        logger.log_info("Basic TRL example completed successfully")
        
    except Exception as e:
        logger.log_error(e, "Basic TRL example failed")
        raise


def ppo_training_example():
    """Example of PPO training with rubric rewards (requires TRL)."""
    
    try:
        from trl import PPOConfig, PPOTrainer
        print("TRL is available - running PPO example")
    except ImportError:
        print("TRL not installed. Install with: pip install trl")
        print("Skipping PPO training example")
        return
    
    # Set up logging
    logger = OpenRubricLogger(log_file="ppo_training.log")
    training_logger = TrainingLogger("ppo_dialogue_training", logger)
    
    try:
        # Load model and tokenizer
        model_name = "microsoft/DialoGPT-small"  # Small model for demo
        
        logger.log_info(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create rubric and reward function
        rubric_path = create_sample_dialogue_rubric()
        
        # Create PPO trainer with rubric
        ppo_trainer = create_ppo_trainer_with_rubric(
            model=model,
            tokenizer=tokenizer,
            rubric_path=rubric_path,
            provider="openai",
            # PPO config
            batch_size=4,
            mini_batch_size=2,
            learning_rate=1e-5,
        )
        
        # Prepare training data
        training_data = prepare_training_data()
        queries = [item["query"] for item in training_data]
        
        logger.log_info("Starting PPO training...")
        
        # Training loop
        for epoch in range(3):  # Small number for demo
            logger.log_info(f"Epoch {epoch + 1}/3")
            
            # Generate responses
            query_tensors = [
                tokenizer.encode(query, return_tensors="pt")[0] 
                for query in queries
            ]
            
            # Generate responses using the model
            response_tensors = []
            with torch.no_grad():
                for query_tensor in query_tensors:
                    response = model.generate(
                        query_tensor.unsqueeze(0),
                        max_length=query_tensor.shape[0] + 50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    response_tensors.append(response[0][query_tensor.shape[0]:])
            
            # Decode responses
            responses = [
                tokenizer.decode(response, skip_special_tokens=True)
                for response in response_tensors
            ]
            
            # Perform PPO step
            stats = ppo_trainer.step(queries, responses)
            
            # Log statistics
            training_logger.log_step(
                reward=stats.get("reward/mean", 0.0),
                additional_metrics={
                    "loss": stats.get("ppo/loss/total", 0.0),
                    "kl_divergence": stats.get("objective/kl", 0.0)
                }
            )
            
            logger.log_info(f"Epoch {epoch + 1} stats: {stats}")
        
        # Save the model
        model_save_path = "./fine_tuned_dialogue_model"
        ppo_trainer.save_model(model_save_path)
        logger.log_info(f"Model saved to: {model_save_path}")
        
        training_logger.end_session()
        
        # Test the fine-tuned model
        logger.log_info("Testing fine-tuned model...")
        
        test_query = "How do I stay motivated?"
        inputs = tokenizer.encode(test_query, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        logger.log_info(f"Test query: {test_query}")
        logger.log_info(f"Generated response: {response}")
        
        logger.log_info("PPO training example completed successfully")
        
    except Exception as e:
        logger.log_error(e, "PPO training failed")
        raise


def supervised_fine_tuning_with_rubric():
    """Example of supervised fine-tuning with rubric evaluation."""
    
    # Set up logging
    logger = OpenRubricLogger(log_file="sft_training.log")
    
    try:
        # Load model and tokenizer
        model_name = "microsoft/DialoGPT-small"
        
        logger.log_info(f"Loading model for SFT: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create rubric reward function for evaluation
        rubric_path = create_sample_dialogue_rubric()
        reward_function = create_trl_reward_function(
            rubric_path=rubric_path,
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Prepare evaluation queries
        eval_queries = [
            "What's a good way to learn programming?",
            "How do I stay healthy?",
            "What's the meaning of life?"
        ]
        
        # Create rubric callback for evaluation during training
        rubric_callback = RubricCallback(
            reward_function=reward_function,
            eval_queries=eval_queries,
            eval_interval=50  # Evaluate every 50 steps
        )
        
        # Prepare training data (simplified)
        training_data = prepare_training_data()
        
        # Create dataset
        def tokenize_function(examples):
            # Combine query and response for causal LM training
            texts = [f"Query: {q} Response: {r}" for q, r in zip(examples["query"], examples["response"])]
            return tokenizer(texts, truncation=True, padding=True, max_length=128)
        
        # Convert to Hugging Face dataset format
        try:
            from datasets import Dataset
            
            dataset = Dataset.from_list(training_data)
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./sft_dialogue_model",
                num_train_epochs=1,
                per_device_train_batch_size=2,
                logging_steps=10,
                eval_steps=50,
                save_steps=100,
                logging_dir="./logs",
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                callbacks=[rubric_callback]
            )
            
            logger.log_info("Starting supervised fine-tuning...")
            
            # Train
            trainer.train()
            
            # Save model
            trainer.save_model("./sft_dialogue_model_final")
            logger.log_info("SFT model saved")
            
        except ImportError:
            logger.log_info("datasets library not available - skipping dataset creation")
            logger.log_info("Install with: pip install datasets")
        
        logger.log_info("Supervised fine-tuning example completed")
        
    except Exception as e:
        logger.log_error(e, "Supervised fine-tuning failed")
        raise


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Set it before running:")
        print("export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    print("Running TRL examples...")
    
    # Run basic example
    print("\n1. Basic TRL reward function test")
    basic_trl_example()
    
    # Run PPO example (if TRL is available)
    print("\n2. PPO training with rubric rewards")
    ppo_training_example()
    
    # Run SFT example
    print("\n3. Supervised fine-tuning with rubric evaluation")
    supervised_fine_tuning_with_rubric()
    
    print("\nAll TRL examples completed!")