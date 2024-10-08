from enum import Enum
from typing import Tuple, Optional, Union
# from loguru import logger

import numpy as np
from mindspore import nn, ops
from mindnlp.transformers import GPT2LMHeadModel, GPT2Config
from mindnlp.core.nn import functional as F
from mindnlp.transformers import BertTokenizer
from mindspore import Tensor
from mindformers import AutoConfig, AutoModel

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')



class MappingType(Enum):
    MLP = 'mlp'
    BERT = 'bert'


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

class GPT2ForSummarization(GPT2LMHeadModel):

    def __init__(self, config, tokenizer):
        super(GPT2ForSummarization, self).__init__(config)
        self.tokenizer = tokenizer
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        labels = None,
    ):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        # Flatten the tokens
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=self.tokenizer.pad_token_id)
        return (loss,)

# class BertMapper(nn.Module):
#     def __init__(self, bert_config, clip_size, prefix_size, prefix_len, constant_len):
#         super(BertMapper, self).__init__()
#         self.prefix_len = prefix_len
#         self.prefix_size = prefix_size
#         self.constant_len = constant_len
#         self.bert = BertModel(config=bert_config)
#         self.linear = nn.Linear(clip_size, prefix_len * prefix_size)
#         self.prefix_const = nn.Parameter(torch.randn(constant_len, prefix_size), requires_grad=True)

#     def forward(self, x):
#         bs = x.size(0)
#         # 将bs个图片向量映射成[bs, prefix_len, prefix_size]
#         prefix = self.linear(x).view(-1, self.prefix_len, self.prefix_size)
#         # [bs, constant_len, prefix_size]
#         constant = self.prefix_const.unsqueeze(0).expand(bs, self.constant_len, self.prefix_size)
#         # 将prefix向量与constant向量拼接，作为bert模型的输入
#         prefix = torch.cat((prefix, constant), dim=1)
#         # 输出捕获attention之后的prefix向量的输出
#         out = self.bert(inputs_embeds=prefix)
#         out = out.last_hidden_state[:, self.prefix_len:]
#         return out


class ClipCaptionModel(nn.Cell):

    def __init__(self, gpt2_path, bert_config,tokenizer, prefix_len=10, clip_size=1024, mapping_type: MappingType = MappingType.MLP,
                 finetune_gpt2=False, constant_len=10):
        super(ClipCaptionModel, self).__init__()

        # 生成模型
        # todo 修改
        # gpt2_config = GPT2Config(vocab_size=len(tokenizer), bos_token_id)
        gpt2_config = GPT2Config(vocab_size=len(tokenizer), bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.pad_token_id)

        # gpt_config = AutoConfig.from_pretrained("gpt2")
        try:
            # self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_path)
            # self.gpt2 = AutoModel.from_pretrained("gpt2")  # 自动加载预置权重到网络中
            self.gpt2 = GPT2ForSummarization(gpt2_config, tokenizer)
            print("succeed to load pretrain gpt2 model")
            # logger.info('succeed to load pretrain gpt2 model')
        except:
            pass
            # config = GPT2Config.from_pretrained(gpt2_path)
            # self.gpt2 = GPT2LMHeadModel.from_config(config)
            # logger.info('random initialize gpt2 model')
        # self.gpt2 = AutoModelWithLMHead.from_pretrained(gpt2_path)
        # 将每个图片向量[clip_size] -> [prefix_len, prefix_size]
        # self.prefix_size = self.gpt2.config.n_embd
        self.prefix_size = 768
        self.prefix_len = prefix_len
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((clip_size, (self.prefix_size * prefix_len) // 2, self.prefix_size * prefix_len))
            print("We are using MLP")
        else:
            print("We are using BERT")
        # else:
        #     self.clip_project = BertMapper(bert_config, clip_size, self.prefix_size, prefix_len, constant_len)
        self.finetune_gpt2 = finetune_gpt2

    def construct(self, clip_embeds, caption_ids, mask):
        """

        :param clip_embeds: 图像embedding, [bs, clip_size]
        :param caption_ids: caption的文本id, [bs, len]
        :param mask: 对于caption的文本id的attention mask, [bs, len]
        :return:
        """
        # caption_inputs_embeds:[bs, caption_len, prefix_size]
        caption_embeds = self.gpt2.transformer.wte(caption_ids)
        print(caption_embeds.shape)

        # prefix_embeds:[bs, prefix_len, prefix_size]
        prefix_embeds = self.clip_project(clip_embeds).view(-1, self.prefix_len, self.prefix_size)
        print(prefix_embeds.shape)
        # embedding_cat:[bs, prefix_len+caption_len, prefix_size]
        embedding_cat = ops.concat((prefix_embeds, caption_embeds), axis=1)
        print(embedding_cat.shape)
        # out = self.gpt2(input_ids=embedding_cat, attention_mask=mask)
        out = self.gpt2.generate(input_ids=embedding_cat)
        print(out.shape)
        # logits:[bs, prefix_len+caption_len, prefix_size]
        logits = out.logits
        return logits

    def parameters(self, recurse: bool = True):
        if self.finetune_gpt2:
            return super(ClipCaptionModel, self).parameters()
        else:
            return self.clip_project.parameters()

    # def train(self, mode: bool = True):
    #     self.set_train(mode)


if __name__ == '__main__':
    model = ClipCaptionModel("","")
    print(model(Tensor([1.0] * 1024),Tensor([1.0] * 10).unsqueeze(0), Tensor(np.ones([1,20]).astype(np.float32))))