import json
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

name = 'llava-9-epoch'
gen_path = Path(f'../out/pororo/{name}/test_images')
data_path = Path('../data/pororo_png')
id_path = data_path / 'train_seen_unseen_ids.npy'
following_path = data_path  / 'following_cache4.npy'
description_path = data_path / 'descriptions.npy'
background_path = data_path / 'background_prompt_test.json'
video_len = 4
samples = 24
use_background = True

def sample_image(im):
    shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
    video_len = int(longer/shorter)
    se = np.random.randint(0, video_len, 1)[0]
    return im.crop((0, se * shorter, shorter, (se+1)*shorter))

_, _, test_ids = np.load(id_path, allow_pickle=True)
test_ids = np.sort(test_ids)[:samples]
followings = np.load(following_path)
descriptions_original = np.load(description_path, allow_pickle=True, encoding='latin1').item()
background_prompts = json.load(open(background_path))
wandb.init(project='pororo', name=name)

for i, src_img_id in tqdm(enumerate(test_ids), total=len(test_ids)):

    wandb_images = []
    for j in range(video_len):
        # load images
        image_path = gen_path / f'gen_sample_{i}_{j}.png'
        image = Image.open(image_path)
        tgt_img_path = str(followings[src_img_id][j])[2:-1]
        tgt_image = sample_image(Image.open(data_path / tgt_img_path))
        image = image.resize(tgt_image.size)

        # load captions
        tgt_img_id = str(tgt_img_path).replace(str(data_path), '').replace('.png', '')
        caption = descriptions_original[tgt_img_id][0].strip()
        background = background_prompts[str(src_img_id)][j]

        wandb_images.append(wandb.Image(image, caption=caption))
        if use_background:
            wandb_images.append(wandb.Image(tgt_image, caption=background))
        else:
            wandb_images.append(wandb.Image(tgt_image, caption='ground truth'))

    wandb.log({str(src_img_id): wandb_images})