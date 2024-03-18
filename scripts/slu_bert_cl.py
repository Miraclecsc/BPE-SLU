# coding=utf8
import sys
import os
import time
import json
import gc
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR

install_path = os.path.abspath(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.vocab import PAD
from utils.batch_bert import from_example_list
from utils.example_bert import Example
from utils.initialization import *
from utils.args import init_args
from model.slu_bert_tagging import SLUTaggingBERT, SLUTaggingBERT_CRF


# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >=
      0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, args.train_path)
dev_path = os.path.join(args.dataroot, 'development.json')
model_name = args.model_name
Example.configuration(args.dataroot, train_path=train_path,
                      word2vec_path=args.word2vec_path, tokenizer_name=model_name)

if not args.testing:
    # train_asr_dataset = Example.load_dataset(train_path)
    train_dataset = Example.load_dataset(train_path)

dev_dataset = Example.load_dataset(dev_path)

print("Load dataset and database finished, cost %.4fs ..." %
      (time.time() - start_time))
# print("Dataset size: train -> %d ; dev -> %d" %
#       (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.num_acts = Example.label_vocab.num_acts
args.num_slots = Example.label_vocab.num_slots
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

# 方便后续添加模型
models = [SLUTaggingBERT, SLUTaggingBERT_CRF]
model = models[args.models](args).to(device)

if args.testing:
    checkpoint = torch.load(open('model.bin', 'rb'), map_location=device)
    model.load_state_dict(checkpoint['model'])
    print("Load saved model from root path")


def set_scheduler(optimizer, args):
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return scheduler


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = AdamW(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(
                args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(
                Example.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path, testing=True)
    predictions = []
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            current_batch = from_example_list(
                args, cur_dataset, device, train=False)
            pred = model.decode(Example.label_vocab, current_batch)

            for idx, p in enumerate(pred[0]):
                ex = current_batch[idx].ex
                for his in p:
                    ex['pred'].append(his.split('-'))

                if ex['utt_id'] != 1:
                    previous_item = predictions.pop(-1)
                    previous_item.append(ex)
                    predictions.append(previous_item)
                else:
                    predictions.append([ex])

    outputs = json.dumps(predictions, indent=4, ensure_ascii=False)

    with open(os.path.join(args.dataroot, 'test.json'), 'w') as wf:
        print(outputs, file=wf)


def compute_difficulty(batch):
    return batch.shape[1]


if not args.testing:
    num_training_steps = (
        (len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    scheduler = set_scheduler(optimizer, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print("Start training ...")
    for i in range(args.max_epoch):
        start_time = time.time()

        # 根据当前epoch决定使用哪个数据集
        if i < 5:
            current_dataset = train_dataset
        else:
            current_dataset = train_dataset
            np.random.shuffle(train_index)

        epoch_loss = 0
        epoch_sep_loss = 0
        epoch_tag_loss = 0
        model.train()
        count = 0
        for j in range(0, nsamples, step_size):
            # 输出当前batch第一个utt的id
            # print(train_dataset[train_index[j]].ex['manual_transcript'])
            cur_dataset = [train_dataset[k]
                           for k in train_index[j: j + step_size]]
            current_batch = from_example_list(
                args, cur_dataset, device, train=True)
            _, loss, sep_loss, tag_loss = model(current_batch)
            if (epoch_sep_loss):
                epoch_sep_loss += sep_loss.item()
            if (epoch_tag_loss):
                epoch_tag_loss += tag_loss.item()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f\tSep Loss: %.4f\tTag Loss: %.4f' % (
            i, time.time() - start_time, epoch_loss / count, epoch_sep_loss / count, epoch_tag_loss / count))
        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' %
              (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('model.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' %
                  (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' %
          (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    predict()
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" %
          (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
