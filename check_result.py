#-----------------------------------------------
#这个脚本是为了检查得出的submission的结果
#显示前25个的图片
#-----------------------------------------------

import json
import os
import shutil
from tqdm import tqdm
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

IMAGE_SIZE = 256                              #每张小图片的大小
IMAGE_ROW = 1                                 #图片行数
NUM_QUERY = 25                                 #显示查询的图片数量
IMAGE_COLUMN = NUM_QUERY + 1                  #26张图片
query_path = 'data/AmurTiger/query/'
save_result_path = 'check/'                   #保存图片的根目录

# 图片拼接
def image_compose(query_id, ans_ids):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))      # 新建一张图片打底
    first_pic = query_path + query_id  # 查询图片的路径
    from_image = Image.open(first_pic).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)  # 打开查询图片并且resize

    draw = ImageDraw.Draw(from_image)                                                     # 写上文字
    draw.text((5,5), query_id, (255, 0, 0),font=ImageFont.truetype('LiberationSans-Regular.ttf', 20))

    to_image.paste(from_image, (0, 0))  # 查询图片粘贴

    for x, ans in enumerate(ans_ids[:NUM_QUERY]):
        ans = str(ans).zfill(6) + '.jpg'
        x = x + 1
        second_pic = query_path + ans  # 查询到的图片
        from_image = Image.open(second_pic).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        draw = ImageDraw.Draw(from_image)  # 写上文字
        draw.text((5, 5), ans, (255, 0, 0),font=ImageFont.truetype('LiberationSans-Regular.ttf', 20))
        to_image.paste(from_image, (x * IMAGE_SIZE, 0))

    os.mkdir(save_result_path + query_id.split('.')[0])
    to_image.save(save_result_path + query_id.split('.')[0] + '/' + 'result.jpg')

with open('submition.json') as f:
    load_dict = json.load(f)
    for r in tqdm(load_dict[:5]):
        query_id = str(r['query_id']).zfill(6) + '.jpg'
        ans_ids = r['ans_ids']

        #拼接结果图片
        image_compose(query_id, ans_ids)

        #粘贴图片
        shutil.copyfile(query_path + query_id, save_result_path + query_id.split('.')[0] + '/' + query_id)
        for pic in ans_ids[:NUM_QUERY]:
            pic = str(pic).zfill(6) + '.jpg'
            shutil.copyfile(query_path + pic, save_result_path + query_id.split('.')[0] + '/' + pic)


