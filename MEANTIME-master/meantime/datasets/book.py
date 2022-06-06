from .base import AbstractDataset

import pandas as pd
import time
import datetime as dt
from datetime import date
from dotmap import DotMap
import pickle


class Book(AbstractDataset):
    @classmethod
    def code(cls):
        return 'book'

    @classmethod
    def url(cls):
        return 'https://ecampus.kookmin.ac.kr/pluginfile.php/1310235/mod_ubboard/attachment/448646/book_transactions.csv?forcedownload=1'

    @classmethod
    def zip_file_content_is_folder(cls):
        return False

    def preprocess(self):
        # 데이터 저장 경로
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return

        # 데이터가 경로에 없으면, 폴더 생성
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        # 데이터 다운
        self.maybe_download_raw_dataset()
        # open dataset
        df = self.load_ratings_df()
        # 데이터의 rating이 기준 이하인 경우 버림
        #df = self.make_implicit(df)
        # 일정 갯수 이하인 데이터 버림(user, item)
        df = self.filter_triplets(df)

        # umap, smap : 유저, 아이템 라벨 인코딩
        # df : 라벨인코딩 된 데이터
        df, umap, smap , side1map, side2map, side3map = self.densify_index(df)
        user2dict, train_targets, validation_targets, test_targets = self.split_df(df, len(umap))


        special_tokens = DotMap()
        special_tokens.pad = 0
        item_count = len(smap)
        special_tokens.mask = item_count + 1
        special_tokens.sos = item_count + 2
        special_tokens.eos = item_count + 3
        num_days = df.days.max() + 1
        num_ratings = len(df)

        dataset = {'user2dict': user2dict,
                'train_targets': train_targets,
                'validation_targets': validation_targets,
                'test_targets': test_targets,
                'umap': umap,
                'smap': smap,
                'special_tokens': special_tokens,
                'num_ratings': num_ratings,
                'side1map' : side1map,
                'side2map' : side2map,
                'side3map' : side3map,
                'num_days': num_days}
            
        with dataset_path.open('wb') as f:
           pickle.dump(dataset, f)
           
    def densify_index(self, df):
        print('Densifying index')
        # user, item 라벨 인코딩(중복값 제거)
        umap = {u: (i+1) for i, u in enumerate(set(df['uid']))}
        smap = {s: (i+1) for i, s in enumerate(set(df['sid']))}

        side1map = {u: (i+1) for i, u in enumerate(set(df['side1']))}
        side2map = {u: (i+1) for i, u in enumerate(set(df['side2']))} 
        side3map = {u: (i+1) for i, u in enumerate(set(df['side3']))} 
        # user-item matrix 생성
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)

        df['side1'] = df['side1'].map(side1map)
        df['side2'] = df['side2'].map(side2map)
        df['side3'] = df['side3'].map(side3map)
        return df, umap, smap, side1map, side2map, side3map

    def split_df(self, df, user_count):
        # timestamp순으로 정렬 하는 함수
        def sort_by_time(d):
            d = d.sort_values(by='timestamp')
            return {'items': list(d.sid), 'timestamps': list(d.timestamp), 'days': list(d.days),
          'side1s' : list(d.side1), 'side2s' : list(d.side2), 'side3s' : list(d.side3)}

        # 최초 구매기록
        min_date = date.fromtimestamp(df.timestamp.min())
        # 데이터에서 최초 구매일자 대비 며칠이 지났는지
        df['days'] = df.timestamp.map(lambda t: (date.fromtimestamp(t) - min_date).days)
        user_group = df.groupby('uid')
        # user2dict : uid별 구매 item
        user2dict = user_group.progress_apply(sort_by_time)
        
        if self.args.split == 'leave_one_out':
            train_ranges = []
            val_positions = []
            test_positions = []
            for user, d in user2dict.items():
                n = len(d['items'])
                train_ranges.append((user, n-2))  # exclusive range
                val_positions.append((user, n-2))
                test_positions.append((user, n-1))
            train_targets = train_ranges
            validation_targets = val_positions
            test_targets = test_positions
        else:
            raise ValueError

        return user2dict, train_targets, validation_targets, test_targets

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('book_transactions.csv')
        df = pd.read_csv(file_path, encoding = 'cp949')
        
        df = df[['회원번호' , '책제목' ,  '일자', '카테고리' , '작가' , '출판사']]
        df.rename(columns = {'회원번호': 'uid' , '책제목' :'sid' , '일자':'timestamp',
                       '카테고리' : 'side1' , '작가' : 'side2' , '출판사' : 'side3'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d')
        df['timestamp']  =df['timestamp'].apply(lambda x : x.timestamp())
        return df
