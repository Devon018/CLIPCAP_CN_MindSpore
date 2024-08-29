from typing import Tuple
import mindspore
from os.path import join
import os
import pickle
import glob
from PIL import Image
# 定义可迭代数据集对象
class ClipCapDataset:
    def __init__(
        self,
        clip_data_path,
        prefix_len,
        tokenizer,
        max_len,
        mode="train",
        normalize_prefix=False,
    ):
        assert mode in ["train", "test"]
        self.normalize_prefix = normalize_prefix
        pad_id = tokenizer.pad_token_id
        if mode == "train":
            save_path = join(os.path.dirname(clip_data_path), "train.pkl")
        else:
            save_path = join(os.path.dirname(clip_data_path), "test.pkl")

        # 加载pkl文件中的数据
        if os.path.isfile(save_path):
            with open(save_path, "rb") as f:
                self.clip_embeds, self.caption_ids_list, self.mask_list = pickle.load(f)
            # logger.info('num of training data'.format(len(self.clip_embeds)))
        else:
            # logger.info('loading dataset:{}'.format(clip_data_path))
            with open(clip_data_path, "rb") as f:
                caption_list, image_id2embed = pickle.load(f)
            # logger.info('num of image embedding:{}'.format(len(image_id2embed)))
            # logger.info('num of captions:{}'.format(len(caption_list)))

            clip_embeds = []
            caption_ids_list = []
            mask_list = []
            for image_id, caption in caption_list:
                clip_embed = image_id2embed[image_id].squeeze(0).float()
                caption_ids = tokenizer.encode(caption, add_special_tokens=False)
                # caption_ids.append(tokenizer.sep_token_id)
                # print(tokenizer.sep_token_id)
                # truncate
                caption_ids = caption_ids[: max_len - prefix_len]
                # print(caption_ids)
                mask = [1] * (prefix_len + len(caption_ids))

                # padding
                padding_len = max_len - prefix_len - len(caption_ids)
                caption_ids += [pad_id] * padding_len
                
                mask += [0] * padding_len

                caption_ids = mindspore.tensor(caption_ids).long()
                mask = mindspore.tensor(mask).long()

                clip_embeds.append(clip_embed)
                caption_ids_list.append(caption_ids)
                mask_list.append(mask)
            with open(save_path, "wb") as f:
                pickle.dump([clip_embeds, caption_ids_list, mask_list], f)
            self.clip_embeds = clip_embeds
            self.caption_ids_list = caption_ids_list
            self.mask_list = mask_list
            # logger.info('num of training data'.format(len(self.clip_embeds)))

    def __len__(self) -> int:
        return len(self.caption_ids_list)

    def __getitem__(self, index: int) -> Tuple[mindspore.Tensor, ...]:
        clip_embed = self.clip_embeds[index]
        caption_ids = self.caption_ids_list[index]
        mask = self.mask_list[index]
        if self.normalize_prefix:
            clip_embed = clip_embed / clip_embed.norm(2, -1)  # todo check
        return clip_embed, caption_ids, mask

class ImageDataset:

    def __init__(self, image_path):
        if os.path.isdir(image_path):
            with open(image_path,"rb") as f:
                image_names, image_id2embed = pickle.load(f)
    
        else:
            print("Cannot find image path.")
        self.image_names = image_names
        self.image_id2embed = image_id2embed
    
    def __len__(self) ->int:
        return len(self.image_names)
    
    def __getitem__(self, index: int) -> Tuple[mindspore.Tensor,...]:
        image_name = self.image_names[index]
        image_embedding  = self.image_id2embed[index]
        return image_name,image_embedding
    
    

print("finish loading")