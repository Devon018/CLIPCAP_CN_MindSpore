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
from mindnlp.transformers import BertTokenizer
from tqdm import tqdm
from dataset import ClipCapDataset
from models.model import ClipCaptionModel
import time
from os.path import join
import os
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/clip_caption.pkl')
    parser.add_argument('--gpt2_path', default='pretrain_models/gpt2')
    parser.add_argument('--bert_path', default='pretrain_models/bert')
    parser.add_argument('--output_path', default='output')
    parser.add_argument("--lr", type=float, default=2e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--prefix_len', type=int, default=10)
    parser.add_argument('--constant_len', type=int, default=10)
    parser.add_argument('--clip_size', type=int, default=512)
    parser.add_argument('--bs_train', type=int, default=2)
    parser.add_argument('--dev_size', type=int, default=10)
    parser.add_argument('--bs_eval', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
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
    model.train()
   # logger.info("start training")
    device = args.device
    for epoch in range(args.epochs):
        #logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx + 1
            clip_embeds, caption_ids, mask = data
            clip_embeds = clip_embeds.to(device).float()
            caption_ids = caption_ids.to(device)
            mask = mask.to(device)
            logits = model(clip_embeds, caption_ids, mask)

            # 计算loss
            shift_logits = logits[..., args.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = caption_ids.view(-1)
            loss = mindspore.nn.loss.SoftmaxCrossEntropyWithLogits(shift_logits, shift_labels)

            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()

            if step % args.eval_step == 0:
                dev_loss = evaluate(args, model, dev_dataloader)
                #writer.add_scalar('loss', dev_loss, step)
                #logger.info('loss at step {} is {}'.format(step, dev_loss.item()))
                model.train()

            if step % args.save_step == 0:
                #logger.info('saving checkpoint at step {}'.format(step))
                save_path = join(args.output_path, 'checkpoint-{}.pt'.format(step))
                mindspore.save_checkpoint(model.state_dict(), save_path)


def evaluate(args, model, dataloader):
    """
    计算数据集上的指标
    :return:
    """
    #model.eval()
    model.set_train(False)
    device = args.device
    #logger.info("Running evaluation")
    eval_loss = 0.0  #
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    for data in tqdm(dataloader):
        clip_embeds, caption_ids, mask = data
        clip_embeds = clip_embeds.to(device).float()
        caption_ids = caption_ids.to(device)
        mask = mask.to(device)
        logits = model(clip_embeds, caption_ids, mask)

        # 计算loss
        shift_logits = logits[..., args.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = caption_ids.view(-1)
        loss = criterion(shift_logits, shift_labels)
        eval_loss += loss.asnumpy().item()
    return eval_loss


def main(args):
    # 分词器
    #tokenizer = BertTokenizer.from_pretrained(args.gpt2_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # BertMapper的模型配置
    #bert_config = BertConfig.from_pretrained(args.bert_path)
    # 加载模型
    model = ClipCaptionModel(
        "", "",tokenizer, args.prefix_len, args.clip_size, args.mapping_type,
        args.finetune_gpt2, args.constant_len
    ).to(args.device)

    if args.do_train:
        # 加载数据集
        dataset = ClipCapDataset(args.data_path, args.prefix_len, tokenizer, args.max_len, 'train', args.normalize_prefix)
       # writer = SummaryWriter(args.output_path)
        #train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [len(dataset) - args.dev_size, args.dev_size])
        #train_dataloader = DataLoader(train_dataset, batch_size=args.bs_train, shuffle=True, num_workers=args.num_workers)
        #dev_dataloader = DataLoader(dev_dataset, batch_size=args.bs_eval, shuffle=True, num_workers=args.num_workers)
        train_dataset, dev_dataset = dataset.split([len(dataset) - args.dev_size, args.dev_size])
        train_dataloader = GeneratorDataset(train_dataset, column_names=['clip_embeds', 'caption_ids', 'mask'],
                                            shuffle=True, batch_size=args.bs_train,
                                            num_parallel_workers=args.num_workers)
        dev_dataloader = GeneratorDataset(dev_dataset, column_names=['clip_embeds', 'caption_ids', 'mask'],
                                          shuffle=True, batch_size=args.bs_eval, num_parallel_workers=args.num_workers)
        t_total = len(train_dataloader) * args.epochs
        lr_scheduler =nn.WarmUpLR(args.lr,args.warmup_steps)
        optimizer = Adam(params=model.parameters(), learning_rate=lr_scheduler)
        train(model, train_dataloader, dev_dataloader, optimizer, args)
    # if args.do_test:
    #     # 加载数据集
    #     test_dataset = ClipCapDataset(args.data_path, args.prefix_len, tokenizer, args.max_len, 'test',
    #                                    args.normalize_prefix)
    #     test_dataloader = DataLoader(test_dataset, batch_size=args.bs_test, shuffle=False,
    #                                  num_workers=args.num_workers)


if __name__ == '__main__':
    args = set_args()
    #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="GPU" if context.get_context("device_target") == "GPU" else "CPU")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        log_file = os.path.join(args.output_path, f'train-{cur_time}.log')
        #logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
        #logger.info(args)
        #writer = SummaryWriter(args.output_path)
        summary_collector = SummaryCollector(log_dir=args.output_path)
    main(args)
