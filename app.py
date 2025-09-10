import gradio as gr
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io

load_dotenv()

pipe = None
stop_inference = False

# Fancy styling
fancy_css = """
#main-container {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #45a049;
}
.gr-slider input {
    color: #4CAF50;
}
.gr-chat {
    font-size: 16px;
}
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    if image is None:
        return None
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Save to bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    
    # Encode to base64
    return base64.b64encode(image_bytes).decode('utf-8')

def prepare_messages_with_images(history, system_message, current_message, current_image):
    """Prepare messages array with proper image handling"""
    messages = [{"role": "system", "content": system_message}]
    
    # Add history messages
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append(msg)
        elif isinstance(msg, list) and len(msg) == 2:
            # Handle gradio chat format [user_msg, bot_msg]
            user_msg, bot_msg = msg
            if user_msg:
                # Check if user message contains images (gradio format)
                if isinstance(user_msg, dict) and "files" in user_msg:
                    content = [{"type": "text", "text": user_msg.get("text", "")}]
                    for file_path in user_msg["files"]:
                        try:
                            with Image.open(file_path) as img:
                                img_b64 = encode_image_to_base64(img)
                                content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                                })
                        except Exception as e:
                            print(f"Error processing image {file_path}: {e}")
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": user_msg})
            
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
    
    # Add current message with image
    content = []
    if current_message:
        content.append({"type": "text", "text": current_message})
    
    if current_image is not None:
        img_b64 = encode_image_to_base64(current_image)
        if img_b64:
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
    
    if content:
        if len(content) == 1 and content[0]["type"] == "text":
            messages.append({"role": "user", "content": content[0]["text"]})
        else:
            messages.append({"role": "user", "content": content})
    
    return messages

def respond(
    message,
    history: list,
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
    use_local_model: bool,
):
    global pipe

    # Get current image from the multimodal input
    current_image = None
    current_text = message
    
    # Handle multimodal input format
    if isinstance(message, dict):
        current_text = message.get("text", "")
        files = message.get("files", [])
        if files:
            try:
                current_image = Image.open(files[0])  # Use first image
            except Exception as e:
                print(f"Error loading image: {e}")

    response = ""

    if use_local_model:
        print("[MODE] local - SmolVLM")
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            
            if pipe is None:
                model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
                pipe = {
                    "processor": AutoProcessor.from_pretrained(model_name),
                    "model": AutoModelForVision2Seq.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                }

            # Prepare prompt for SmolVLM with proper image token handling
            if current_image is not None:
                # For SmolVLM, we need to include an image token in the prompt when an image is present
                prompt = f"{system_message}\n"
                for msg in history[-3:]:  # Keep last 3 exchanges for context
                    if isinstance(msg, list) and len(msg) == 2:
                        user_msg, bot_msg = msg
                        if user_msg:
                            prompt += f"User: {user_msg}\n"
                        if bot_msg:
                            prompt += f"Assistant: {bot_msg}\n"
                
                # Add current message with image token
                prompt += f"User: <image>\n{current_text}\nAssistant:"
                
                # Process with image
                inputs = pipe["processor"](
                    text=prompt,
                    images=current_image,
                    return_tensors="pt"
                )
            else:
                # Text-only mode
                prompt = f"{system_message}\n"
                for msg in history[-5:]:  # Keep more context for text-only
                    if isinstance(msg, list) and len(msg) == 2:
                        user_msg, bot_msg = msg
                        if user_msg:
                            prompt += f"User: {user_msg}\n"
                        if bot_msg:
                            prompt += f"Assistant: {bot_msg}\n"
                
                prompt += f"User: {current_text}\nAssistant:"
                
                # Process text-only
                inputs = pipe["processor"](
                    text=prompt,
                    return_tensors="pt"
                )

            # Move inputs to the same device as model
            device = next(pipe["model"].parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = pipe["model"].generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 512),  # Limit tokens for stability
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=pipe["processor"].tokenizer.eos_token_id,
                    eos_token_id=pipe["processor"].tokenizer.eos_token_id,
                )

            # Decode response
            generated_text = pipe["processor"].decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Clean up the response
            response = generated_text.strip()
            # Remove any leftover prompt text
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            yield response

        except Exception as e:
            yield f"Error with local model: {str(e)}. Please make sure transformers and torch are installed."

    else:
        print("[MODE] API - Qwen2.5-VL")
        
        if hf_token is None or not getattr(hf_token, "token", None):
            yield "⚠️ Please log in with your Hugging Face account first."
            return

        try:
            client = InferenceClient(token=hf_token.token, model="Qwen/Qwen2.5-VL-7B-Instruct")
            
            # Prepare messages for API
            messages = prepare_messages_with_images(history, system_message, current_text, current_image)

            for chunk in client.chat_completion(
                messages,
                max_tokens=max_tokens,
                stream=True,
                temperature=temperature,
                top_p=top_p,
            ):
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        response += delta.content
                        yield response

        except Exception as e:
            yield f"Error with API model: {str(e)}"


# Create the multimodal chatbot interface
chatbot = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a helpful AI assistant that can understand both text and images.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Checkbox(label="Use Local Model (SmolVLM-256M)", value=False),
    ],
    type="messages",
    multimodal=True,  # Enable multimodal support
)

with gr.Blocks(css=fancy_css) as demo:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'>🌟 Multimodal AI Chatbot 🖼️</h1>")
        gr.LoginButton()
    
    gr.Markdown("""
    ### Features:
    - 💬 **Text Chat**: Ask questions and have conversations
    - 🖼️ **Image Understanding**: Upload images and ask questions about them
    - 🌐 **API Mode**: Uses Qwen2.5-VL-7B-Instruct (requires HF login)
    - 🖥️ **Local Mode**: Uses SmolVLM-256M-Instruct (runs locally)
    """)
    
    chatbot.render()

if __name__ == "__main__":
    demo.launch()