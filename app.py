import gradio as gr
from huggingface_hub import InferenceClient
import os
import torch
from dotenv import load_dotenv
import base64
from PIL import Image
import io

load_dotenv()

pipe = None
stop_inference = False

def initialize_local_model():
    """Initialize SmolVLM model at startup"""
    global pipe
    print("🚀 Starting SmolVLM initialization...")
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch
        
        model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
        print(f"📦 Loading {model_name}...")
        
        processor = AutoProcessor.from_pretrained(model_name)
        print("✅ Processor loaded")
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("✅ Model loaded")
        
        pipe = {
            "processor": processor,
            "model": model
        }
        
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"🎯 SmolVLM ready on {device}!")
        
    except ImportError as e:
        print(f"⚠️  Missing dependencies for local model: {e}")
        print("💡 Install with: pip install transformers torch torchvision")
        pipe = None
    except Exception as e:
        print(f"❌ Error loading SmolVLM: {e}")
        pipe = None

# Initialize the local model at startup
print("🔄 Initializing local model in background...")
initialize_local_model()

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
            # Already in proper format
            messages.append(msg)
        elif isinstance(msg, list) and len(msg) == 2:
            # Handle gradio chat format [user_msg, bot_msg]
            user_msg, bot_msg = msg
            
            # Process user message
            if user_msg is not None:
                user_content = None
                
                if isinstance(user_msg, dict):
                    # Gradio multimodal format: {"text": "...", "files": [...]}
                    text_part = user_msg.get("text", "").strip()
                    files_part = user_msg.get("files", [])
                    
                    if files_part:
                        # Has images
                        content_parts = []
                        if text_part:
                            content_parts.append({"type": "text", "text": text_part})
                        
                        for file_path in files_part:
                            try:
                                with Image.open(file_path) as img:
                                    img_b64 = encode_image_to_base64(img)
                                    if img_b64:
                                        content_parts.append({
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                                        })
                            except Exception as e:
                                print(f"Error processing history image {file_path}: {e}")
                        
                        if content_parts:
                            user_content = content_parts
                    else:
                        # Text only from dict
                        if text_part:
                            user_content = text_part
                
                elif isinstance(user_msg, str):
                    # Simple text message
                    if user_msg.strip():
                        user_content = user_msg.strip()
                
                # Add user message if we have content
                if user_content:
                    messages.append({"role": "user", "content": user_content})
            
            # Process bot response
            if bot_msg is not None and isinstance(bot_msg, str) and bot_msg.strip():
                messages.append({"role": "assistant", "content": bot_msg.strip()})
    
    # Add current message with image
    if current_message or current_image is not None:
        current_content = []
        
        # Add text if provided
        if current_message and current_message.strip():
            current_content.append({"type": "text", "text": current_message.strip()})
        
        # Add image if provided
        if current_image is not None:
            img_b64 = encode_image_to_base64(current_image)
            if img_b64:
                current_content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
        
        # Add current message in appropriate format
        if current_content:
            if len(current_content) == 1 and current_content[0]["type"] == "text":
                # Text only - use simple string format
                messages.append({"role": "user", "content": current_content[0]["text"]})
            else:
                # Multimodal - use array format
                messages.append({"role": "user", "content": current_content})
    
    return messages

def respond(
    message,
    history: list,
    system_message,
    max_tokens,
    temperature,
    top_p,
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
        
        if pipe is None:
            yield "❌ Local model not available. Please check the console for initialization errors."
            return
            
        try:
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
            yield f"Error with local model: {str(e)}"

    else:
        print("[MODE] API - Qwen2.5-VL")
        
        # Get token from environment variable
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            yield "⚠️ HF_TOKEN not found in environment variables. Please set it in .env file."
            return

        try:
            client = InferenceClient(token=hf_token, model="Qwen/Qwen2.5-VL-7B-Instruct")
            
            # Simplified message preparation for API - avoid complex history processing
            messages = [{"role": "system", "content": system_message}]
            
            # Only include recent text-only history to avoid format issues
            for msg in history[-3:]:  # Keep only last 3 exchanges
                if isinstance(msg, list) and len(msg) == 2:
                    user_msg, bot_msg = msg
                    
                    # Only add simple text messages from history to avoid format issues
                    if isinstance(user_msg, str) and user_msg.strip():
                        messages.append({"role": "user", "content": user_msg.strip()})
                        if isinstance(bot_msg, str) and bot_msg.strip():
                            messages.append({"role": "assistant", "content": bot_msg.strip()})
            
            # Add current message
            if current_text or current_image is not None:
                current_content = []
                
                if current_text and current_text.strip():
                    current_content.append({"type": "text", "text": current_text.strip()})
                
                if current_image is not None:
                    img_b64 = encode_image_to_base64(current_image)
                    if img_b64:
                        current_content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        })
                
                if current_content:
                    if len(current_content) == 1 and current_content[0]["type"] == "text":
                        messages.append({"role": "user", "content": current_content[0]["text"]})
                    else:
                        messages.append({"role": "user", "content": current_content})
            
            # Debug: Print what we're sending to API
            print("DEBUG: Messages being sent to API:")
            for i, msg in enumerate(messages):
                print(f"  [{i}] Role: {msg['role']}")
                if isinstance(msg['content'], str):
                    print(f"      Content: {msg['content'][:100]}...")
                elif isinstance(msg['content'], list):
                    print(f"      Content: {len(msg['content'])} parts")
                    for j, part in enumerate(msg['content']):
                        print(f"        [{j}] Type: {part.get('type', 'unknown')}")
                        if part.get('type') == 'text':
                            print(f"            Text: {part.get('text', '')[:50]}...")
                        elif part.get('type') == 'image_url':
                            print(f"            Image: base64 data present")

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
            print(f"DEBUG: Full error details: {e}")
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
    
    gr.Markdown(f"""
    ### Features:
    - 💬 **Text Chat**: Ask questions and have conversations
    - 🖼️ **Image Understanding**: Upload images and ask questions about them
    - 🌐 **API Mode**: Uses Qwen2.5-VL-7B-Instruct (requires HF token in .env)
    - 🖥️ **Local Mode**: Uses SmolVLM-256M-Instruct (preloaded at startup)
    
    ### Status:
    - 🤖 **Local Model**: {'✅ Ready' if pipe is not None else '❌ Not Available'}
    - 🔑 **API Token**: {'✅ Configured' if os.getenv('HF_TOKEN') else '❌ Missing'}
    """)
    
    chatbot.render()

if __name__ == "__main__":
    demo.launch(share=True)
