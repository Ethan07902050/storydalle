import os, re
import csv
import nltk
import pickle
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch.utils.data
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import json
import pickle as pkl

class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, tokenizer, preprocess, mode='train', video_len=4, out_img_folder='', return_labels=False):
        self.lengths = []
        self.followings = []
        self.images = []
        self.img_dataset = ImageFolder(img_folder)
        self.img_folder = img_folder
        self.labels = np.load(os.path.join(img_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        self.video_len = video_len
        self.descriptions_original = np.load(os.path.join(img_folder, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
        # self.descriptions = np.load(os.path.join(img_folder, 'descriptions_vec.npy'), allow_pickle=True, encoding='latin1').item() # used in the eccv camera-ready version
        self.descriptions = pkl.load(open(os.path.join(img_folder, 'descriptions_vec_512.pkl'), 'rb'))

        if os.path.exists(os.path.join(img_folder, 'img_cache4.npy')) and os.path.exists(os.path.join(img_folder, 'following_cache4.npy')):
            self.images = np.load(os.path.join(img_folder, 'img_cache4.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(img_folder, 'following_cache4.npy'))
            self.counter = ''
        else:
            counter = np.load(os.path.join(img_folder, 'frames_counter.npy'), allow_pickle=True).item()
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Counting total number of frames")):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(self.img_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name] - (self.video_len-1):
                    continue
                following_imgs = []
                for i in range(self.video_len-1):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(self.img_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(self.img_folder, 'img_cache4.npy'), self.images)
            np.save(os.path.join(self.img_folder, 'following_cache4.npy'), self.followings)

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        if mode == 'train':
            self.ids = np.sort(train_ids)
        elif mode =='val':
            self.ids = np.sort(val_ids)
        elif mode =='test':
            self.ids = np.sort(test_ids)
        else:
            raise ValueError

        with open(os.path.join(img_folder, f'background_llama_{mode}.json')) as f:
            self.background_prompts = json.load(f)

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.return_labels = return_labels
        self.out_img_folder = out_img_folder

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, item):

        src_img_id = self.ids[item]


        src_img_path = os.path.join(self.img_folder, str(self.images[src_img_id])[2:-1])
        tgt_img_paths = [str(self.followings[src_img_id][i])[2:-1] for i in range(self.video_len)]
        # print(src_img_path, tgt_img_path)

        # open the source images
        src_image = self.preprocess(self.sample_image(Image.open(src_img_path).convert('RGB')))

        # open the target image and caption
        tgt_img_ids = [str(tgt_img_path).replace(self.img_folder, '').replace('.png', '') for tgt_img_path in tgt_img_paths]

        if self.out_img_folder:
            tgt_images = [self.preprocess(Image.open(os.path.join(self.out_img_folder, 'gen_sample_%s_%s.png' % (item, frame_idx))).convert('RGB')) for frame_idx in range(self.video_len)]
        else:
            tgt_images = [self.preprocess(self.sample_image(Image.open(os.path.join(self.img_folder, tgt_img_path)).convert('RGB'))) for tgt_img_path in tgt_img_paths]
        # image = Image.open(os.path.join(self.out_dir, 'img-' + str(item) + '.png')).convert('RGB')

        # sometimes llama does not generate exactly 4 background descriptions
        prompts = self.background_prompts[str(src_img_id)]
        l = len(prompts)
        if l != self.video_len:
            prompts += [prompts[-1]] * (self.video_len - l) 

        captions = []
        for tgt_img_id, prompt in zip(tgt_img_ids, prompts):
            caption = self.descriptions_original[tgt_img_id][0] + ' ' + prompt
            captions.append(caption)

        if self.tokenizer is not None:
            tokens = [self.tokenizer.encode(caption.lower()) for caption in captions]
            tokens = torch.stack([torch.LongTensor(token.ids) for token in tokens])
        else:
            tokens = captions

        sentence_embeddings = [torch.tensor(self.descriptions[tgt_img_id][0]) for tgt_img_id in tgt_img_ids]

        if self.return_labels:
            labels = [torch.tensor(self.labels[img_id]) for img_id in tgt_img_ids]
            return torch.stack(tgt_images), torch.stack(labels), tokens, src_image, torch.stack(sentence_embeddings)
        else:
            return torch.stack(tgt_images), tokens, src_image, torch.stack(sentence_embeddings)

    def __len__(self):
        return len(self.ids)