import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_and_tokenizer(model_name="google/gemma-3-1b-it", cache_dir=None):
    """
    Downloads and initializes the Gemma 3 model and tokenizer from Hugging Face.
    
    Args:
        model_name (str): The name of the model on Hugging Face Hub
        cache_dir (str, optional): Directory to store the downloaded model
        
    Returns:
        tuple: (tokenizer, model) The initialized tokenizer and model
    """
    print(f"Loading model: {model_name}")
    
    # Set cache directory if provided
    if cache_dir is None:
        # Default to a subdirectory in the current directory
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download and load the tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Download and load the model
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
        device_map="auto",  # Automatically determine device placement
        cache_dir=cache_dir
    )
    
    print("Model and tokenizer loaded successfully!")
    return tokenizer, model

def generate_text(tokenizer, model, prompt, max_new_tokens=450, temperature=0.7, top_p=0.9):
    """
    Generate text using the loaded model.
    
    Args:
        tokenizer: The model tokenizer
        model: The language model
        prompt (str): The input prompt for text generation
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness (higher = more random)
        top_p (float): Controls diversity via nucleus sampling
        
    Returns:
        str: The generated text
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# Example usage
if __name__ == "__main__":
    # Test the model loading and text generation
    tokenizer, model = get_model_and_tokenizer()
    
    test_prompt = "Explain the importance of mental health awareness in three sentences."
    print("\nTest prompt:", test_prompt)
    
    response = generate_text(tokenizer, model, test_prompt)
    print("\nGenerated response:")
    print(response)