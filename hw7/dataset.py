import json
from torch.utils import data
from transformers import BertTokenizerFast, AutoTokenizer
import opencc


class BaseDataset(data.Dataset):

    def __init__(self,
                 file: str,
                 pretrained: str,
                 maxlen: int,
                 mode: str = 'train',
                 to_cn: bool = False) -> None:
        super().__init__()
        data = self.load_json(file)
        converter = opencc.OpenCC('t2s.json')
        self.questions = data['questions']
        self.context = data['paragraphs']
        if to_cn:
            for q in self.questions:
                q['question_text'] = converter.convert(q['question_text'])
                q['answer_text'] = converter.convert(q['answer_text'])
            self.context = [converter.convert(q) for q in self.context]
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.maxlen = maxlen
        self.mode = mode

        print(f'split: {self.mode}, num_samples: {len(self.questions)}')

    @staticmethod
    def load_json(path: str):
        with open(path) as f:
            return json.load(f)

    def __len__(self):
        return len(self.questions)

    def tokenize(self, a, b, **kwargs):
        return self.tokenizer(a,
                              b,
                              padding='max_length',
                              max_length=self.maxlen,
                              truncation='longest_first',
                              return_tensors='pt',
                              **kwargs)

    @classmethod
    def dataloader(cls,
                   file: str,
                   pretrained: str,
                   maxlen: int,
                   mode: str = 'train',
                   batch_size=32,
                   **kwargs) -> data.DataLoader:
        ds = cls(file, pretrained, maxlen, mode, **kwargs)
        dl = data.DataLoader(ds,
                             batch_size=batch_size,
                             num_workers=10,
                             drop_last=mode != 'test',
                             shuffle=mode == 'train')
        return dl


class QADataset(BaseDataset):

    def __init__(self,
                 file: str,
                 pretrained: str,
                 maxlen: int,
                 mode: str = 'train',
                 **kwargs) -> None:
        super().__init__(file, pretrained, maxlen, mode, **kwargs)

    def __getitem__(self, index):
        question = self.questions[index]
        context = self.context[question['paragraph_id']]
        print(question)
        start = question['answer_start']
        end = question['answer_end'] + 1 # [, )

        inputs = self.tokenize(
            question['question_text'],
            context,
            return_offsets_mapping=True,
            stride=128,
            return_overflowing_tokens=True,
        )

        if self.mode == 'test':
            return question['id'], inputs.input_ids[0], inputs.token_type_ids[
                0], inputs.attention_mask[0]

        offset = inputs.pop('offset_mapping')[0]
        sequence_ids = inputs.sequence_ids(0)
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        if offset[context_start][0] > end or offset[context_end][1] < start:
            start_pos = end_pos = 0
        else:
            start_pos = context_start
            while start_pos <= context_end and offset[start_pos][0] <= start:
                start_pos += 1
            start_pos -= 1

            end_pos = context_end
            while end_pos >= context_start and offset[end_pos][1] >= end:
                end_pos -= 1
            end_pos += 1

        return inputs.input_ids[0], inputs.token_type_ids[
            0], inputs.attention_mask[0], start_pos, end_pos


if __name__ == '__main__':
    from tqdm import tqdm
    # dl = get_multiple_choice_dataloader('data',
    #                                     'train',
    #                                     'ckiplab/bert-base-chinese',
    #                                     maxlen=128,
    #                                     batch_size=2)
    dl = QADataset.dataloader('data/hw7_train.json', 'ckiplab/bert-base-chinese', 512, 'train', 8, to_cn=True)
    for d in tqdm(dl):
        # input_ids, _, _, start, end = d
        pass
        # print(ds.tokenizer.decode(input_ids[start:end+1]))