import gradio as gr
import json
import time
import os
from openai_infer import APIInfer
from config import DEEPSEEK_API_KEY, BASE_URL, MODEL
import text2img_workflow_infer as text2img
import img2img_workflow_infer as img2img

# 加载工作流文件
text2img_workflow_path = "workflow\\AnimeAPI.json"
with open(text2img_workflow_path, 'r', encoding="utf-8") as f:
    text2img_workflow_data = json.load(f)
    
# 加载图生图工作流文件
img2img_workflow_path = "workflow\\image2imageAPI.json"
with open(img2img_workflow_path, 'r', encoding="utf-8") as f:
    img2img_workflow_data = json.load(f)

# 初始化API推理
LLM_url = BASE_URL
api_key = DEEPSEEK_API_KEY
model_name = MODEL
url = "http://127.0.0.1:8188"
apiinfer = APIInfer(url=LLM_url, api_key=api_key, model_name=model_name)

# 系统提示词
system_prompt = "你是一个极其优秀的分词器翻译机器，请将输入的句子进行分词并翻译成英文词汇,不要任何其他的回复。"

# 用户提示词模板
user_prompt_template = """
你是一个极其优秀的分词器翻译机器，请将输入的句子进行分词并翻译成英文词汇,不要任何其他的回复。
下面将提供几个例子:
第一个例子:
用户输入:一位独自出现的女孩，她有着一头白色的短发，整齐的齐刘海垂在额前，灰白的眼眸正平静地注视着观众，双唇轻抿，神情淡然。她的指尖指甲或黑或白，透着几分独特的风格，耳间缀着精致的耳环，手指上戴着戒指，腿上还纹有细腻的图案，周身点缀着小巧的花朵，与她身上穿着的白色连衣裙相呼应，整体萦绕着纯净的白色主题。女孩以双腿交叉的姿势坐着，部分身体浸在水中，水波轻轻环绕着她的肢体，她手中稳稳握着剑鞘，身旁还放置着配套的长剑。画面采用富有张力的动态角度呈现，细致的光线精准勾勒出她的身形轮廓与水中的涟漪，些许 chromatic aberration（色差）效果增添了独特的视觉质感，无论是人物细节的刻画还是整体氛围的营造，都达到了杰作级的最佳质量，每一处细节都尽显精致。

你将其翻译成英文，并返回结果。你返回的结果如下:
1girl, solo,weapon, solo, sword, sheath, holding sheath, white hair, jewelry, sitting, tattoo, black nails, bangs, flower, white eyes, blunt bangs, earrings, looking at viewer, white dress, dress, crossed legs, ring, grey eyes, water, medium hair, breasts, leg tattoo, fingernails, white theme, closed mouth, white nails, short hair, partially submerged, chromatic aberration, detailed lighting, dynamic angle,
,masterpiece,best quality

第二个例子:
用户输入:初音未来她有着白色瞳孔头发被风吹起，手里拿着麦克风，周身萦绕着能量元素。整体是鲜明的蓝色调，搭配抽象背景，透着梦幻感，线条细腻，画面质量达到杰作级的最佳水准
你将其翻译成英文，并返回结果。你返回的结果如下:
1girl, hatsune miku, white pupils, power elements, microphone, vibrant blue color palette, abstract,abstract background, dreamlike atmosphere, delicate linework, wind-swept hair, energy
,masterpiece,best quality,

第三个例子:
用户输入:独自出镜的可爱女孩，粉色长发垂落，不齐的刘海与修长侧发修饰脸庞，宇宙紫色眼眸泛着轮廓光，她侧脸朝向一旁却望向观众，下垂的眼尾带着浅笑，还微微歪着头。红色环形日食光晕环绕身旁，红色项圈搭配精致的紫色水手服，胸前大红领巾格外亮眼，她拱着背，指尖捧着发光的星星。从下方以荷兰角度拍摄她的上半身肖像，逆光与轮廓光勾勒出身形，彩色光粒子在周身灵动飘散。背景是梦幻的宇宙夜空，极光在深色天幕中流转，细节丰富的奇幻场景与模糊虚化的前景形成对比，景深与体积光让画面更具层次，整体以 4K 高分辨率呈现，细节极致丰富，尽显高审美与杰作级的优质质感。
你将其翻译成英文，并返回结果。你返回的结果如下:
masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, 1girl, solo, cute, pink hair, long hair, choppy bangs, long sidelocks, nebulae cosmic purple eyes, rimlit eyes, facing to the side, looking at viewer, downturned eyes, light smile, red annular solar eclipse halo, red choker, detailed purple serafuku, big red neckerchief, fingers, glowing stars in hand, arched back, from below, dutch angle, portrait, upper body, head tilt, colorful, rim light, backlit, (colorful light particles:1.2), cosmic sky, aurora, chaos, perfect night, fantasy background, BREAK, detailed background, blurry foreground, bokeh, depth of field, volumetric lighting


不要任何其他任何的中文回复,只需要返回英文单词即可,需要详细大量的细致的英文单词进行详细描述。描述应符合ComfyUI提示词工作流的规范，请勿返回其他任何中文内容。
下面是用户的输入:

{}
"""

# 默认的负面提示词
default_negative_prompt = "worst quality, low quality, lowres, normal quality, monochrome, grayscale, skin spots, acnes, skin blemishes, age spot, ugly, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, blurry, bad anatomy, bad proportions, extra limbs, disfigured, missing arms, extra legs, fused fingers, too many fingers, unclear eyes, lowers, bad hands, missing fingers, extra digit, bad feet, wrong feet, extra feets, text, watermark, error, extra digit, fewer digits, cropped, bad proportion, out of frame, duplicate, bad art, disabled, deformed, distorted, malformation, amputation, missing body, extra head, poorly drawn face, disfigured, bad eyes, deformed eye, cross-eyed, bad tongue, malformed limbs, extra arms, missing arms, extra legs, bad legs, many legs, more than two legs, strange fingers, fused hands, connected hand, wrong fingers, 4 fingers, 3 fingers"

# 文生图函数
def generate_text2img(description, width=832, height=1216):
    # 调用LLM进行翻译
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_template.format(description)}
    ]
    
    response = apiinfer.infer(messages=messages, stream=False)
    prompt_positive = response.choices[0].message.content
    
    # 生成随机种子
    random_seed = int(time.time())
    
    # 调用工作流生成图像
    text2img.inference(url, text2img_workflow_data, prompt_positive, default_negative_prompt, random_seed, width, height)
    
    # 获取最新生成的图像
    output_dir = "./output/"
    image_files = [f for f in os.listdir(output_dir) if f.startswith("ComfyUI_")]
    if image_files:
        latest_image = max(image_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
        return os.path.join(output_dir, latest_image), prompt_positive
    return None, None

# 图生图函数
def generate_img2img(input_image, description, width=832, height=1216):
    # 调用LLM进行翻译
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_template.format(description)}
    ]
    
    response = apiinfer.infer(messages=messages, stream=False)
    prompt_positive = response.choices[0].message.content
    
    # 生成随机种子
    random_seed = int(time.time())
    
    # 调用工作流生成图像
    img2img.inference(url, img2img_workflow_data, input_image, prompt_positive, default_negative_prompt, random_seed, width, height)
    
    # 获取最新生成的图像
    output_dir = "./output/"
    image_files = [f for f in os.listdir(output_dir) if f.startswith("ComfyUI_")]
    if image_files:
        latest_image = max(image_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
        return os.path.join(output_dir, latest_image), prompt_positive
    return None, None

# 创建Gradio界面
with gr.Blocks(title="AI图像生成器") as demo:
    gr.Markdown("# AI图像生成器")
    gr.Markdown("输入中文描述，系统将自动翻译并生成相应的图像")
    
    # 添加界面切换按钮
    with gr.Row():
        with gr.Column(scale=1):
            text2img_btn_switch = gr.Button("文生图", variant="primary", scale=1)
        with gr.Column(scale=1):
            img2img_btn_switch = gr.Button("图生图", variant="secondary", scale=1)
    
    # 文生图界面
    with gr.Row(visible=True) as text2img_interface:
        with gr.Column(scale=2):
            # 输入区域
            text_input = gr.Textbox(label="图像描述", placeholder="请输入中文描述...", lines=10)
            with gr.Row():
                width_slider = gr.Slider(minimum=512, maximum=4048, value=832, step=64, label="宽度")
                height_slider = gr.Slider(minimum=512, maximum=4048, value=1216, step=64, label="高度")
            submit_btn = gr.Button("生成图像")
        
        with gr.Column(scale=3):
            # 输出区域
            image_output = gr.Image(label="生成的图像")
            prompt_output = gr.Textbox(label="生成的提示词", lines=5)
    
    # 图生图界面
    with gr.Row(visible=False) as img2img_interface:
        with gr.Column(scale=2):
            img2img_input = gr.Image(label="上传参考图像", type="filepath")
            img2img_description = gr.Textbox(label="图像描述", placeholder="请输入中文描述...", lines=10)
            with gr.Row():
                img2img_width = gr.Slider(minimum=512, maximum=4048, value=832, step=64, label="宽度")
                img2img_height = gr.Slider(minimum=512, maximum=4048, value=1216, step=64, label="高度")
            img2img_btn = gr.Button("生成图像")
        
        with gr.Column(scale=3):
            img2img_output = gr.Image(label="生成的图像")
            img2img_prompt = gr.Textbox(label="生成的提示词", lines=5)
    
    # 界面切换逻辑
    def switch_to_text2img():
        return gr.update(visible=True), gr.update(visible=False), gr.update(variant="primary"), gr.update(variant="secondary")
    
    def switch_to_img2img():
        return gr.update(visible=False), gr.update(visible=True), gr.update(variant="secondary"), gr.update(variant="primary")
    
    text2img_btn_switch.click(
        fn=switch_to_text2img,
        inputs=[],
        outputs=[text2img_interface, img2img_interface, text2img_btn_switch, img2img_btn_switch]
    )
    
    img2img_btn_switch.click(
        fn=switch_to_img2img,
        inputs=[],
        outputs=[text2img_interface, img2img_interface, text2img_btn_switch, img2img_btn_switch]
    )
    
    # 文生图处理函数
    def process_text2img(description, width, height):
        image_path, prompt = generate_text2img(description, width, height)
        return image_path, prompt
    
    # 图生图处理函数
    def process_img2img(input_image, description, width, height):
        if not input_image:
            return None, "请先上传参考图像。"
        image_path, prompt = generate_img2img(input_image, description, width, height)
        return image_path, prompt
    
    # 设置提交按钮事件
    submit_btn.click(fn=process_text2img, inputs=[text_input, width_slider, height_slider], outputs=[image_output, prompt_output])
    img2img_btn.click(fn=process_img2img, inputs=[img2img_input, img2img_description, img2img_width, img2img_height], outputs=[img2img_output, img2img_prompt])

if __name__ == "__main__":
    # 启动Gradio界面
    demo.launch()