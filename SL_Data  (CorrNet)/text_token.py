# from Tokenizer import TextTokenizer
# import torch

# tokenizer = {}
# tokenizer['pretrained_model_name_or_path'] = 'mBart_zh'
# tokenizer['pruneids_file']='mBart_zh/old2new_vocab.pkl'
# tokenizer['tgt_lang'] = 'zh_CN'
# text_tokenizer = TextTokenizer(tokenizer)
# a = ['您买的每一元彩票，都将有两角钱捐献给山区的孩子。']
# # 1654, 2504,   43, 3292,  676, 4821,   32,  298,  415,  117, 1448, 2646, 2156,  625, 5057, 1087,  529,  819, 3979,   38,    2,    1
# print(text_tokenizer(a))

import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from Tokenizer import GlossTokenizer_G2T, TextTokenizer
import pickle, math

translation_network = {
    "GlossEmbedding": {
        "freeze": False,
        "gloss2embed_file": "mBart_zh/gloss_embeddings.bin"
    },
    "GlossTokenizer": {
        "gloss2id_file": "mBart_zh/gloss2ids.pkl",
        "src_lang": "zh_CSL"
    },
    "TextTokenizer": {
        "pretrained_model_name_or_path": "mBart_zh",
        "pruneids_file": "mBart_zh/old2new_vocab.pkl",
        "tgt_lang": "zh_CN"
    },
    "load_ckpt": "phoenix-2014T_g2t/best.ckpt",
    "pretrained_model_name_or_path": "mBart_zh",
    "overwrite_cfg": {
        "attention_dropout": 0.1,
        "dropout": 0.3
    }
}

class TranslationNetwork(torch.nn.Module):
    def __init__(self, input_type='feature', cfg=None, task='S2T') -> None:
        super().__init__()
        self.task = task
        self.input_type = input_type
        self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg["TextTokenizer"])
        self.model = MBartForConditionalGeneration.from_pretrained(
            cfg['pretrained_model_name_or_path'],
                **cfg.get('overwrite_cfg', {})
        )
        self.input_dim = self.model.config.d_model
        self.input_embed_scale = math.sqrt(self.model.config.d_model)
        self.gloss_tokenizer = GlossTokenizer_G2T(tokenizer_cfg=cfg['GlossTokenizer'])
        self.gloss_embedding = self.build_gloss_embedding(**cfg['GlossEmbedding'])

    def build_gloss_embedding(self, gloss2embed_file, from_scratch=False, freeze=False):
        gloss_embedding = torch.nn.Embedding(
            num_embeddings=len(self.gloss_tokenizer.id2gloss),
            embedding_dim=self.model.config.d_model,
            padding_idx=self.gloss_tokenizer.gloss2id['<pad>'])
        if from_scratch:
            assert freeze == False
        else:
            gls2embed = torch.load(gloss2embed_file)
            print(len(gls2embed))
            self.gls2embed = gls2embed
            with torch.no_grad():
                # print(self.gloss_tokenizer.id2gloss.items())
                for id_, gls in self.gloss_tokenizer.id2gloss.items():
                    if gls in gls2embed:
                        assert gls in gls2embed, gls
                        gloss_embedding.weight[id_, :] = gls2embed[gls]
        return gloss_embedding

    def forward(self, input):
        output_dict = self.gloss_embedding(input)
        return output_dict

SLT = TranslationNetwork(cfg=translation_network)
tensor_id = torch.tensor([14,2], dtype=torch.long)
a = SLT(tensor_id)
print(a.shape)


