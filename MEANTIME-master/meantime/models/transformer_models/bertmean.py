from .bert_base import BertBaseModel
from .embeddings import *
from .bodies import BertBody
from .bodies import MeantimeBody
from .heads import *

import torch.nn as nn


class BertMeanModel(BertBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info

        hidden = args.hidden_units
        self.output_info = args.output_info
        absolute_kernel_types = args.absolute_kernel_types
        relative_kernel_types = args.relative_kernel_types

        
        ### 토큰 임베딩
        self.token_embedding = TokenEmbedding(args)
        ### 포지션 임베딩
        self.positional_embedding = PositionalEmbedding(args)
        ### 사이드 임베딩
        self.side_embedding1 = SideEmbedding1(args)
        self.side_embedding2 = SideEmbedding2(args)
        self.side_embedding3 = SideEmbedding3(args)

      
        ### 타임스탬프 임베딩
        self.absolute_kernel_embeddings_list = nn.ModuleList()
        if absolute_kernel_types is not None and len(absolute_kernel_types) > 0:
            for kernel_type in absolute_kernel_types.split('-'):
                if kernel_type == 'p':  # position
                    emb = PositionalEmbedding(args)
                elif kernel_type == 'd':  # day
                    emb = DayEmbedding(args)
                elif kernel_type == 'c':  # constant
                    emb = ConstantEmbedding(args)
                else:
                    raise ValueError
                self.absolute_kernel_embeddings_list.append(emb)

        self.relative_kernel_embeddings_list = nn.ModuleList()
        if relative_kernel_types is not None and len(relative_kernel_types) > 0:
            for kernel_type in relative_kernel_types.split('-'):
                if kernel_type == 's':  # time difference
                    emb = SinusoidTimeDiffEmbedding(args)
                elif kernel_type == 'e':
                    emb = ExponentialTimeDiffEmbedding(args)
                elif kernel_type == 'l':
                    emb = Log1pTimeDiffEmbedding(args)
                else:
                    raise ValueError
                self.relative_kernel_embeddings_list.append(emb)

        ### 타임스탬프 길이
        self.La = len(self.absolute_kernel_embeddings_list)
        self.Lr = len(self.relative_kernel_embeddings_list)
        self.L = self.La + self.Lr  

        # Sanity check
        assert hidden % self.L == 0, 'multi-head has to be possible'
        assert len(self.absolute_kernel_embeddings_list) > 0 or len(self.relative_kernel_embeddings_list) > 0



        ### 바디
        self.bert_body = BertBody(args)
        self.meantime_body = MeantimeBody(args, self.La, self.Lr)

        ### 헤드
        self.head = BertMeanDotProductPredictionHead(args, self.token_embedding.emb, self.side_embedding1.emb, self.side_embedding2.emb, self.side_embedding3.emb)
        
        ### 정규화
        self.ln = nn.LayerNorm(args.hidden_units)

        ### 드랍아웃
        self.dropout = nn.Dropout(p=args.dropout)

        ### 가중치 초기화
        self.init_weights()

        ### MISC
        self.ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    @classmethod
    def code(cls):
        return 'bertmean'


    
##########################################################################################################
    def get_logits(self, d):
        x = d['tokens']

        attn_mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        ### bert 정의
        e = self.token_embedding(d) + self.positional_embedding(d) + self.side_embedding1(d) + self.side_embedding2(d)+ self.side_embedding3(d) 
        e = self.ln(e)
        e = self.dropout(e)  

        ### meantime 정의
        absolute_kernel_embeddings = [self.dropout(emb(d)) for emb in self.absolute_kernel_embeddings_list]  
        relative_kernel_embeddings = [self.dropout(emb(d)) for emb in self.relative_kernel_embeddings_list]
        token_embeddings = self.dropout(self.token_embedding(d)) 


        info = {} if self.output_info else None
        # bert 바디 받는 부분
        b = self.bert_body(e, attn_mask, info)

        ### meantime 바디 받는 부분
        last_hidden = self.meantime_body(token_embeddings, attn_mask, absolute_kernel_embeddings,
                                relative_kernel_embeddings,
                                info=info)

        b += last_hidden

        return b, info
        

    def get_scores(self, d, logits):
        if self.training:  
            # 헤드 받는 부분
            h = self.head(logits)  
        else: 
            candidates = d['candidates'] 
            h = self.head(logits, candidates)  
        return h

