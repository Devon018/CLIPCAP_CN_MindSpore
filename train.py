import mindspore
from mindspore import nn
from mindspore.nn import Adam
import mindspore.nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
from mindspore import context
from mindspore.train.callback import SummaryCollector
from mindspore.train.serialization import save_checkpoint
from mindspore.dataset import GeneratorDataset
from tqdm import tqdm
from dataset import ClipCapDataset
from model import ClipCaptionModel
import mindnlp
from mindnlp.transformers import BertConfig, BertTokenizer
from mindnlp.transformers import AutoTokenizer
import time
from os.path import join
import os
import argparse
from loguru import logger

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./clip_caption.pkl')
    parser.add_argument('--gpt2_path', default='pretrain_models/gpt2')
    parser.add_argument('--bert_path', default='pretrain_models/bert')
    parser.add_argument('--output_path', default='output')
    parser.add_argument("--lr", type=float, default=2e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--prefix_len', type=int, default=10)
    parser.add_argument('--constant_len', type=int, default=10)
    parser.add_argument('--clip_size', type=int, default=1024)
    parser.add_argument('--bs_train', type=int, default=20)
    parser.add_argument('--dev_size', type=int, default=10)
    parser.add_argument('--bs_eval', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument("--save_step", type=int, default=1000, help="训练多少步，保存一次模型")
    parser.add_argument("--eval_step", type=int, default=100, help="训练多少步,记录一次指标")
    parser.add_argument('--finetune_gpt2', help='finetune gpt2', action='store_true', default=False)
    parser.add_argument('--mapping_type', type=str, default='mlp', choices=['mlp', 'bert'], help='mlp or bert')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument("--do_train", action='store_true', default=True)
    # parser.add_argument("--do_test", action='store_true', default=True)
    args = parser.parse_args()
    return args


#def train(model, train_loader, dev_dataloader, optimizer, scheduler, args):
def train(model, train_loader, dev_dataloader, optimizer, args):
    model.set_train()
    logger.info("start training")
    def forward_nw(clip_embeds,captions_ids,mask):
            logits = model(clip_embeds, caption_ids, mask)

            # 计算loss
            shift_logits = logits[..., args.prefix_len - 1:-1, :].contiguous().view(-1, logits.shape[-1])
            shift_labels = caption_ids.view(-1)
            loss = mindspore.ops.cross_entropy(shift_logits, shift_labels)
            return loss,shift_logits
    # device = args.device
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            
            step = epoch * len(train_loader) + batch_idx + 1
            clip_embeds, caption_ids, mask = data

            grad_fn = mindspore.value_and_grad(forward_nw, None, model.trainable_params(), has_aux=True)
            (loss, _), grads = grad_fn(clip_embeds,caption_ids,mask)
            optimizer(grads)
            loss = float(loss.asnumpy())

        dev_loss = evaluate(args, model, dev_dataloader)
        #writer.add_scalar('loss', dev_loss, step)
        logger.info('loss at epoch {} is {}'.format(epoch, dev_loss.item()))

        if epoch == 0:
            best_dev_loss = dev_loss.item()
            logger.info('saving best checkpoint at step {}'.format(step))
            save_path = join(args.output_path, 'checkpoint-{}.pt'.format(epoch))
            mindspore.save_checkpoint(model, save_path)
        elif dev_loss.item() < best_dev_loss:
            best_dev_loss = dev_loss.item()
            logger.info('saving best checkpoint at step {}'.format(step))
            save_path = join(args.output_path, 'checkpoint-{}.pt'.format(epoch))
            mindspore.save_checkpoint(model, save_path)
        model.set_train()

                


def evaluate(args, model, dataloader):
    """
    计算数据集上的指标
    :return:
    """
    model.set_train(False)
    # device = args.device
    #logger.info("Running evaluation")
    eval_loss = 0.0  #
    for _,data in enumerate(tqdm(dataloader)):
        clip_embeds, caption_ids, mask = data
        logits = model(clip_embeds, caption_ids, mask)

        # 计算loss
        shift_logits = logits[..., args.prefix_len - 1:-1, :].contiguous().view(-1, logits.shape[-1])
        shift_labels = caption_ids.view(-1)
        loss = mindspore.ops.cross_entropy(shift_logits, shift_labels)
        loss = loss.mean()
        eval_loss +=loss
        # eval_loss += loss.asnumpy().item()
    return eval_loss


def main(args):
    # 加载glm3的分词器
    tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2-0.5B-Instruct",mirror='modelscope')

    # 加载模型
    model = ClipCaptionModel(tokenizer, args.prefix_len, args.clip_size, args.constant_len)

    if args.do_train:
        # 加载数据集
        dataset = ClipCapDataset(args.data_path, args.prefix_len, tokenizer, args.max_len, 'train', args.normalize_prefix)

        dataloader =GeneratorDataset(dataset, column_names=['clip_embeds', 'caption_ids', 'mask'],
                                              shuffle=True,num_parallel_workers=args.num_workers)
        dataloader = dataloader.batch(args.bs_train)
        train_dataloader, dev_dataloader = dataloader.split([0.7,0.3])
       
        t_total = len(train_dataloader) * args.epochs
        lr_scheduler =nn.WarmUpLR(args.lr,args.warmup_steps)
        optimizer = Adam(params=model.trainable_params(), learning_rate=lr_scheduler)
        train(model, train_dataloader, dev_dataloader, optimizer, args)
   
if __name__ == '__main__':
    args = set_args()
    #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # context.set_context(mode=context.GRAPH_MODE,
    #                     device_target="GPU" if context.get_context("device_target") == "GPU" else "CPU")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        log_file = os.path.join(args.output_path, f'train-{cur_time}.log')
        #logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
        #logger.info(args)
        #writer = SummaryWriter(args.output_path)
        # summary_collector = SummaryCollector(log_dir=args.output_path)
    main(args)
