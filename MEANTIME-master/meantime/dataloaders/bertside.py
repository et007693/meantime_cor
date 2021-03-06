from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):
    @classmethod
    def code(cls):
        return 'bertside'

    def _get_dataset(self, mode):
        if mode == 'train':
            return self._get_train_dataset()
        elif mode == 'val':
            return self._get_eval_dataset('val')
        else:
            return self._get_eval_dataset('test')

    # get train dataset
    # 학습을 위한 데이터셋 재처리
    def _get_train_dataset(self):
        train_ranges = self.train_targets
        dataset = BertTrainDataset(self.args, self.dataset, self.train_negative_samples, self.rng, train_ranges)
        return dataset

    # get evalutation dataset
    # 평가를 위한 데이터셋 재처리
    def _get_eval_dataset(self, mode):
        positions = self.validation_targets if mode=='val' else self.test_targets
        dataset = BertEvalDataset(self.args, self.dataset, self.test_negative_samples, positions)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    # dataset : dataloader/base에서 나온 최종 data
    def __init__(self, args, dataset, negative_samples, rng, train_ranges):
        self.args = args
        self.user2dict = dataset['user2dict']
        self.users = sorted(self.user2dict.keys())
        self.train_window = args.train_window
        self.max_len = args.max_len
        self.mask_prob = args.mask_prob
        self.special_tokens = dataset['special_tokens']
        self.num_users = len(dataset['umap'])
        self.num_items = len(dataset['smap'])
        self.side1_count = len(dataset['side1map'])
        self.side2_count = len(dataset['side2map'])
        self.side3_count = len(dataset['side3map'])
        self.rng = rng
        self.train_ranges = train_ranges

        self.index2user_and_offsets = self.populate_indices()

        self.output_timestamps = args.dataloader_output_timestamp
        self.output_days = args.dataloader_output_days
        self.output_user = args.dataloader_output_user

        self.negative_samples = negative_samples

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)

    def populate_indices(self):
        index2user_and_offsets = {}
        i = 0
        T = self.max_len
        W = self.train_window

        # offset is exclusive
        # train 학습 범위 안에서 기준 값 이상으로 구매한 유저들의 정보를 다룸
        # user : uid, pos : item 갯수(일정 갯수 이하로 잘린 데이터에서 구매한 갯수)
        for user, pos in self.train_ranges:
            if W is None or W == 0:
                # 구매한 아이템 갯수
                offsets = [pos]
            else:
                # window size로 나누어서 offset 생성
                # 구매한 갯수가 maxlen보다 작은 user는 offset에 pos로 설정
                offsets = list(range(pos, T-1, -W))  # pos ~ T
                if len(offsets) == 0:
                    offsets = [pos]
            for offset in offsets:
                index2user_and_offsets[i] = (user, offset)
                i += 1
        return index2user_and_offsets

    def __len__(self):
        return len(self.index2user_and_offsets)

    # max_len 아이템만 가지고 나머지 아이템은 0으로 처리
    def __getitem__(self, index):
        user, offset = self.index2user_and_offsets[index]
        seq = self.user2dict[user]['items']
        beg = max(0, offset-self.max_len)
        end = offset  # exclude offset (meant to be)
        seq = seq[beg:end]

        seq_1 = self.user2dict[user]['side1s']
        seq_1 = seq_1[beg:end]
        
        seq_2 = self.user2dict[user]['side2s']
        seq_2 = seq_2[beg:end]
        
        seq_3 = self.user2dict[user]['side3s']
        seq_3 = seq_3[beg:end]
        
##########
        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.special_tokens.mask)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        padding_len = self.max_len - len(tokens)
        valid_len = len(tokens)

        tokens = [0] * padding_len + tokens
        labels = [0] * padding_len + labels

        d = {}
        d['tokens'] = torch.LongTensor(tokens)
        d['labels'] = torch.LongTensor(labels)
##########
        tokens = []
        labels = []
        for s in seq_1:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.special_tokens.mask)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.side1_count))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        padding_len = self.max_len - len(tokens)
        valid_len = len(tokens)

        tokens = [0] * padding_len + tokens
        labels = [0] * padding_len + labels

        d['side1'] = torch.LongTensor(tokens)
        d['label1'] = torch.LongTensor(labels)


##########
        tokens = []
        labels = []
        for s in seq_2:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.special_tokens.mask)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.side2_count))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        padding_len = self.max_len - len(tokens)
        valid_len = len(tokens)

        tokens = [0] * padding_len + tokens
        labels = [0] * padding_len + labels
        
        d['side2'] = torch.LongTensor(tokens)
        d['label2'] = torch.LongTensor(labels)

##########
        tokens = []
        labels = []
        for s in seq_3:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.special_tokens.mask)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.side3_count))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        padding_len = self.max_len - len(tokens)
        valid_len = len(tokens)

        tokens = [0] * padding_len + tokens
        labels = [0] * padding_len + labels
        
        d['side3'] = torch.LongTensor(tokens)
        d['label3'] = torch.LongTensor(labels)

        if self.output_timestamps:
            timestamps = self.user2dict[user]['timestamps']
            timestamps = timestamps[beg:end]
            timestamps = [0] * padding_len + timestamps
            d['timestamps'] = torch.LongTensor(timestamps)

        if self.output_days:
            days = self.user2dict[user]['days']
            days = days[beg:end]
            days = [0] * padding_len + days
            d['days'] = torch.LongTensor(days)

        if self.output_user:
            d['users'] = torch.LongTensor([user])

        return d

class BertEvalDataset(data_utils.Dataset):
    def __init__(self, args, dataset, negative_samples, positions):
        self.user2dict = dataset['user2dict']
        self.positions = positions
        self.max_len = args.max_len
        self.num_items = len(dataset['smap'])
        self.side1_count = len(dataset['side1map'])
        self.side2_count = len(dataset['side2map'])
        self.side3_count = len(dataset['side3map'])
        self.special_tokens = dataset['special_tokens']
        self.negative_samples = negative_samples

        self.output_timestamps = args.dataloader_output_timestamp
        self.output_days = args.dataloader_output_days
        self.output_user = args.dataloader_output_user

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        # positions = self.validation_targets if mode=='val' else self.test_targets
        user, pos = self.positions[index]
        # 유저 구매 기록
        seq = self.user2dict[user]['items']

        # max_len(=10)보다 길이가 짧으면 0, 길면 pos - max_len
        beg = max(0, pos + 1 - self.max_len)
        end = pos + 1
        # 구매 길이(for padding)
        seq = seq[beg:end]

        seq_1 = self.user2dict[user]['side1s']
        seq_1 = seq_1[beg:end]
        seq_1 = [0] * padding_len + seq_1
        d['side1'] = torch.LongTensor(seq_1)

        seq_2 = self.user2dict[user]['side2s']
        seq_2 = seq_2[beg:end]
        seq_2 = [0] * padding_len + seq_2
        d['side2'] = torch.LongTensor(seq_2)
        
        
        seq_3 = self.user2dict[user]['side3s']
        seq_3 = seq_3[beg:end]
        seq_3 = [0] * padding_len + seq_3
        d['side3'] = torch.LongTensor(seq_3)


        # negs : negative sample, answer : 정답값(문자), labels : 정답값의 위치
        negs = self.negative_samples[user]
        # seq의 마지막은 정답값
        answer = [seq[-1]]
        # candidiate = 정답 + negative sample값
        candidates = answer + negs
        # 정답값의 위치 labeling
        labels = [1] * len(answer) + [0] * len(negs)

        # 정답값을 masking
        seq[-1] = self.special_tokens.mask
        # padding 길이
        padding_len = self.max_len - len(seq)
        # max_len보다 짧은 seq에 zero padding
        seq = [0] * padding_len + seq

        tokens = torch.LongTensor(seq)
        candidates = torch.LongTensor(candidates)
        labels = torch.LongTensor(labels)
        d = {'tokens':tokens, 'candidates':candidates, 'labels':labels}

        if self.output_timestamps:
            timestamps = self.user2dict[user]['timestamps']
            timestamps = timestamps[beg:end]
            timestamps = [0] * padding_len + timestamps
            d['timestamps'] = torch.LongTensor(timestamps)

        if self.output_days:
            days = self.user2dict[user]['days']
            days = days[beg:end]
            days = [0] * padding_len + days
            d['days'] = torch.LongTensor(days)

        if self.output_user:
            d['users'] = torch.LongTensor([user])

        
        return d
