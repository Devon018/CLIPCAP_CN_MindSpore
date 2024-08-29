from typing import Tuple, Optional, Union
from loguru import logger

import numpy as np
from mindspore import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.transformers import BertTokenizer
from mindspore import Tensor
from mindnlp.transformers import AutoModelForCausalLM,AutoTokenizer

# Load pretrained model and tokenizer

class MLP(nn.Cell):
    def construct(self, x):
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Dense(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.SequentialCell(*layers)

class ClipCaptionModel(nn.Cell):

    def __init__(self,tokenizer, prefix_len=10, clip_size=1024, constant_len=10):
        super(ClipCaptionModel, self).__init__()

        # 生成模型

        self.tokenizer  = tokenizer
        self.language_model = AutoModelForCausalLM.from_pretrained("qwen/Qwen2-0.5B-Instruct",mirror='modelscope')
        logger.info('succeed to load pretrain glm model')
        # 将每个图片向量[clip_size] -> [prefix_len, prefix_size]

        self.prefix_size = 896
        self.prefix_len = prefix_len

        print("We are using MLP")
        self.clip_project = MLP((clip_size, (self.prefix_size*self.prefix_len) // 2, self.prefix_size*self.prefix_len))


    def construct(self, clip_embeds, caption_ids, mask):
        """

        :param clip_embeds: 图像embedding, [bs, clip_size]
        :param caption_ids: caption的文本id, [bs, len]
        :param mask: 对于caption的文本id的attention mask, [bs, len]
        :return:
        """

        prefix_embeds = self.clip_project(clip_embeds).view(-1, self.prefix_len, self.prefix_size)
        wte = self.language_model.get_input_embeddings()
        caption_embeds = wte(caption_ids)


        # prefix_embeds:[bs, prefix_len, prefix_size]
        # embedding_cat:[bs, prefix_len+caption_len, prefix_size]
        embedding_cat = ops.concat((prefix_embeds, caption_embeds), axis=1)
        # prefix_embeds = self.clip_project(clip_embeds)
        # print(prefix_embeds.shape)
        # [batch_size,prefix_len,prefix_size]
        out = self.language_model(inputs_embeds=embedding_cat, attention_mask=mask)
        logits = out.logits
        # print(logits)
        # print(type(logits))
        return logits


if __name__ == '__main__':
    model = ClipCaptionModel("","")
    print(model(Tensor([1.0] * 1024),Tensor([1.0] * 10).unsqueeze(0), Tensor(np.ones([1,20]).astype(np.float32))))