import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import gradio as gr
from mingunivision.mingunivisioninfer import MingUniVisionInfer
import time
import uuid
from PIL import Image
import numpy as np
import socket
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "int8", "int4"], help="推理精度")
args = parser.parse_args()

model = None

# 寻找可用端口
def find_port(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("localhost", port)) == 0:
            print(f"端口 {port} 已被占用，正在寻找可用端口...")
            return find_port(port=port + 1)
        else:
            return port
        
# 保存上传的图像
def save_uploaded_image(image):
    if image is None:
        return None
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    filename = f"upload_{uuid.uuid4().hex}.png"
    image.save(filename)
    return filename

# 判断图像生成
def is_image_generation_request(text):
    text = text.lower()
    return ("生成" in text or "create" in text or "generate" in text) and ("图片" in text or "图像" in text or "image" in text)

# 判断图像编辑
def is_image_edit_request(text):
    text = text.lower()
    edit_keywords = ["编辑", "修改", "change", "edit", "换成", "改成", "替换", "改为", "调整", "变换", "变成", "调整"]
    return any(keyword in text for keyword in edit_keywords)

# 处理用户消息
def process_message(text, image_path=None):
    try:
        # 图像生成
        if is_image_generation_request(text):
            processed_text = text.replace("生成", "").replace("图片", "").replace("图像", "")
            processed_text = "Please generate the corresponding image based on the description. " + processed_text
            messages = [{
                "role": "HUMAN",
                "content": [{"type": "text", "text": processed_text}],
            }]
            output_image_prefix = f"{int(time.time())}"
            model.generate(messages, max_new_tokens=512, output_image_prefix=output_image_prefix)
            image_path = f"{output_image_prefix}.png"
            return image_path, "image", None
        
        # 图像编辑
        elif is_image_edit_request(text) and image_path:
            # 识别编辑区域
            messages = [{
                "role": "HUMAN",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"Given the edit instruction: {text}, please identify the editing region"},
                ],
            }]
            edit_prefix = f"{int(time.time())}"
            model.generate(messages, max_new_tokens=512, for_edit=True, output_image_prefix=edit_prefix+"_edit_1")
            
            # 执行编辑
            messages = [{
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": text},
                ],
            }]
            model.generate(messages, max_new_tokens=512, for_edit=True, output_image_prefix=edit_prefix+"_edit_2")
            
            # 优化结果
            messages = [{
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": "Refine the image for better clarity."},
                ],
            }]
            model.generate(messages, max_new_tokens=512, for_edit=True, output_image_prefix=edit_prefix+"_edit_3")
            
            edited_image_path = f"{edit_prefix}_edit_3.png"
            return edited_image_path, "image", None
        
        # 图像理解
        elif image_path:
            messages = [{
                "role": "HUMAN",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text},
                ],
            }]
            output_text = model.generate(messages, max_new_tokens=512)
            return output_text, "text", None
        
        # 文本对话
        else:
            messages = [{
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": text},
                ],
            }]
            output_text = model.generate(messages, max_new_tokens=512)
            return output_text, "text", None
    
    except Exception as e:
        return f"处理请求时出错: {str(e)}", "text", str(e)
    
    finally:
        model.reset_inner_state()

# 聊天处理
def chat(message, history, image_input, last_image_path):
    global model
    if model is None:
        model = MingUniVisionInfer("./models/Ming-UniVision-16B-A3B", args.dtype)
    if image_input is not None:
        image_path = save_uploaded_image(image_input)
        last_image_path = image_path
    else:
        image_path = last_image_path
    
    response, response_type, _ = process_message(message, image_path)
    
    if response_type == "image" and response:
        history.append((message, (response,)))
        last_image_path = response
    elif response:
        history.append((message, response or "抱歉，我无法生成有效响应"))
    
    return history, None, last_image_path, ""

# 清除聊天
def clear_chat():
    return [], None, ""


with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">Ming-UniVision</h2>
            </div>
            """)
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=800)
            msg = gr.Textbox(placeholder="输入您的消息...")
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空")
        with gr.Column(scale=1):
            image_input = gr.Image(label="上传图像", type="pil")
            gr.Markdown("## 使用指南")
            gr.Markdown("""
            - **图像生成**: 输入"生成一张可爱女孩的图片"
            - **图像理解**: 先上传图片，然后提问
            - **图像编辑**: 先上传图片，然后说"把衣服颜色改成红色"
            """)
    last_image_path = gr.State(value=None)

    gr.on(
        triggers = [msg.submit, submit_btn.click],
        fn = chat, 
        inputs = [msg, chatbot, image_input, last_image_path], 
        outputs = [chatbot, image_input, last_image_path, msg]
    )
    clear_btn.click(clear_chat, None, [chatbot, image_input, msg])


if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=find_port(args.server_port),
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )