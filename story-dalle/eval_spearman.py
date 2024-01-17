import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from scipy import stats

from transformers import CLIPProcessor, CLIPVisionModel
import torch
from torch.nn.functional import cosine_similarity

model_name = "openai/clip-vit-base-patch16"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPVisionModel.from_pretrained(model_name)
model.to('cuda')
np.random.seed(42)

name = 'llava-9-epoch'
gen_path = Path(f'../out/pororo/{name}/test_images')
data_path = Path('../data/pororo_png')
id_path = data_path / 'train_seen_unseen_ids.npy'
embedding_path = data_path / 'reference_clip.pkl'
video_len = 4

def sample_image(im):
    shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
    video_len = int(longer/shorter)
    se = np.random.randint(0, video_len, 1)[0]
    return im.crop((0, se * shorter, shorter, (se+1)*shorter))

def compute_clip_embedding(image):
    image = processor(images=image, return_tensors="pt")
    pixel_values = image.pixel_values.to('cuda')
    with torch.no_grad():
        embedding = model(pixel_values=pixel_values).last_hidden_state.mean(dim=1)
    return embedding

# Calculate similarity between image pairs
def compute_clip_scores(image_embeddings):
    scores = []

    for i in range(len(image_embeddings)):
        for j in range(i + 1, len(image_embeddings)):
            similarity = cosine_similarity(
                image_embeddings[i], 
                image_embeddings[j]
            ).item()
            scores.append(similarity)

    return scores

def load_embeddings(test_ids):
    if not embedding_path.exists():
        following_path = data_path  / 'following_cache4.npy'
        followings = np.load(following_path)

        print('compute target embeddings')
        embeddings = {}
        for src_img_id in tqdm(test_ids):
            tgt_img_paths = [str(followings[src_img_id][i])[2:-1] for i in range(video_len)]
            tgt_images = [sample_image(Image.open(data_path / tgt_img_path)) for tgt_img_path in tgt_img_paths]
            tgt_embeddings = [compute_clip_embedding(image) for image in tgt_images]
            embeddings[src_img_id] = tgt_embeddings

        with open(embedding_path, 'wb') as f:
            pickle.dump(embeddings, f)
    else:
        with open(embedding_path, 'rb') as f: 
            embeddings = pickle.load(f)
    return embeddings

def compute_spearman_correlation(image_embeddings, tgt_embeddings):
    ref_scores = compute_clip_scores(tgt_embeddings)
    pred_scores = compute_clip_scores(image_embeddings)
    res = stats.spearmanr(pred_scores, ref_scores)
    return res.statistic

_, _, test_ids = np.load(id_path, allow_pickle=True)
test_ids = np.sort(test_ids)
tgt_embeddings = load_embeddings(test_ids)

corr = []
for i, src_img_id in tqdm(enumerate(test_ids), total=len(test_ids)):
    images = [Image.open(gen_path / f'gen_sample_{i}_{j}.png') for j in range(video_len)]
    image_embeddings = [compute_clip_embedding(image) for image in images]
    corr.append(compute_spearman_correlation(image_embeddings, tgt_embeddings[src_img_id]))

print('correlation:', np.mean(corr))