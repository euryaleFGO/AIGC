from openai import OpenAI
from config import DEEPSEEK_API_KEY,BASE_URL,MODEL
import os

class APIInfer:
    def __init__(self,url,api_key,model_name):
        self.url = url
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key,base_url=self.url)

    def infer(self,messages,stream=True,temperature=1.9,top_p =1):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            temperature=temperature,
            top_p =top_p,
        )
        return response
    
# url = BASE_URL
# api_key = DEEPSEEK_API_KEY
# model_name = MODEL
# client = OpenAI(api_key=api_key,base_url=url)


if __name__ == "__main__":
    
    prompt = """
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
    
    """
    
    url = BASE_URL
    api_key = DEEPSEEK_API_KEY
    model_name = MODEL
    apiinfer = APIInfer(url=url,api_key=api_key,model_name=model_name)
    
    while True:
        query = input()
        messages = [
            {"role": "system", "content": "你是一个极其优秀的分词器翻译机器，请将输入的句子进行分词并翻译成英文词汇,不要任何其他的回复。"},
            {"role": "user", "content": str(prompt)+str(query)}
        ]

        # result = response.choices[0].message.content
        # print(result)
        response = apiinfer.infer(messages=messages)
        for res in response:
            result = res.choices[0].delta.content
            if result:
                print(result,end="",flush=True)
                
        print("\n")
        