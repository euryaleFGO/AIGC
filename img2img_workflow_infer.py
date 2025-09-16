import requests
import json
import time

def update_workflow(workflow_data,image,prompt_positive,prompt_negative,random_seed,width=832,height=1216):
    workflow_data["21"]["inputs"]["image"] = image
    workflow_data["22"]["inputs"]["text_b"] = prompt_positive
    workflow_data["26"]["inputs"]["text"] = prompt_negative
    workflow_data["29"]["inputs"]["seed"] = random_seed
    workflow_data["31"]["inputs"]["width"] = width
    workflow_data["31"]["inputs"]["height"] = height
    return workflow_data

def workflow_submit(url,work_flow):
    data = {"prompt": work_flow}
    try:
        response = requests.post(url + "/prompt",json=data)
        result = response.json()
        prompt_id = result["prompt_id"]
        print("工作流提交成功,提交id为:",prompt_id)
    except Exception as e:
        print("工作流提交失败,错误信息为:",e)
        prompt_id = None
    return prompt_id


def workflow_status(url, workflow_id):
    printed_running = False  # 防止“执行中”重复打印
    printed_waiting = False  # 防止“等待中”重复打印
    while True:
        try:
            queue = requests.get(url + "/queue")
            queue_result = queue.json()
            # 恢复：获取“正在执行队列”和“等待队列”（queue_pending）
            queue_running = queue_result.get("queue_running", [])
            queue_waiting = queue_result.get("queue_pending", [])  # 恢复等待队列的获取
            exec_info = queue_result.get("exec_info", {})
            queue_remaining = exec_info.get("queue_remaining", 0)

            # 恢复：判断任务是否在“执行队列”或“等待队列”
            is_running = any(item[1] == workflow_id for item in queue_running)
            is_waiting = any(item[1] == workflow_id for item in queue_waiting)  # 恢复等待状态判断

            # 状态打印逻辑：每种状态只打印一次，避免重复刷屏
            if is_running and not printed_running:
                print("当前任务正在执行中……")
                print(f"当前等待队列（queue_pending）内容：{queue_waiting}")  # 可选：打印等待队列具体内容
                printed_running = True
                printed_waiting = False  # 若从等待转为执行，重置等待标记
            elif is_waiting and not printed_waiting:
                print("当前任务正在等待中……")
                print(f"当前等待队列（queue_pending）内容：{queue_waiting}")  # 打印等待队列具体内容
                printed_waiting = True
                printed_running = False  # 若从执行转为等待，重置执行标记
            elif not is_running and not is_waiting and queue_remaining == 0:
                print("任务执行完毕")
                print(f"最终等待队列（queue_pending）内容：{queue_waiting}")  # 可选：打印最终等待队列状态
                return

            time.sleep(1)
        except Exception as e:
            print(f"状态查询失败: {e}")
            return

def workflow_result(url, workflow_id):
    reponse = requests.get(url + f"/history/"+str(workflow_id))
    result = reponse.json()
    images = result[str(workflow_id)]["outputs"]
    print(images)
    image_info_list = images["28"]["images"]
    image_name = image_info_list[0]["filename"]
    image_subfolder = image_info_list[0]["subfolder"]
    image_type = image_info_list[0]["type"]
    print(result)
    response = requests.get(url+"/view", params={"type": image_type, "filename": image_name, "subfolder": image_subfolder})  # 获取图片
    img_result = response.content            # 图片的数据
    with open("./output/" + image_name, "wb") as f:    # 把图片数据保存到本地
        f.write(img_result)
    print("图片保存至本地，文件名为：", image_name)

def inference(url, workflow_data, images,prompt_positive, prompt_negative, random_seed, width=832, height=1216):
    workflow_data = update_workflow(workflow_data, images,prompt_positive, prompt_negative, random_seed, width, height)
    workflow_id = workflow_submit(url, workflow_data)
    if workflow_id:
        workflow_status(url, workflow_id)
        workflow_result(url, workflow_id)


if __name__ == '__main__':
    url = "http://127.0.0.1:8188"
    workflow_path = "workflow/image2imageAPI.json"
    with open(workflow_path, 'r', encoding="utf-8") as f:
        workflow_data = json.load(f)
    images_path = r"example.png"
    prompt_positive = "Cartethyia \(wuthering waves\), Cartethyia teen style clothes, small blue and white crown of thorns, Cartethyia white dress, black forehead mark, 1girl, blonde hair, pointy ears, long hair, blue eyes, braid, cowboy shot, abstract background, water, blue theme, masterpiece,best quality,"

    # prompt_positive = input()

    prompt_negative = "worst quality, low quality, lowres, normal quality, monochrome, grayscale, skin spots, acnes, skin blemishes, age spot, ugly, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, blurry, bad anatomy, bad proportions, extra limbs, disfigured, missing arms, extra legs, fused fingers, too many fingers, unclear eyes, lowers, bad hands, missing fingers, extra digit, bad feet, wrong feet, extra feets, text, watermark, error, extra digit, fewer digits, cropped, bad proportion, out of frame, duplicate, bad art, disabled, deformed, distorted, malformation, amputation, missing body, extra head, poorly drawn face, disfigured, bad eyes, deformed eye, cross-eyed, bad tongue, malformed limbs, extra arms, missing arms, extra legs, bad legs, many legs, more than two legs, strange fingers, fused hands, connected hand, wrong fingers, 4 fingers, 3 fingers"
    random_seed = int(time.time())

    inference(url, workflow_data, images_path,prompt_positive, prompt_negative, random_seed,512,650)


