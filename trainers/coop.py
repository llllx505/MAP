import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import json



_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # print(x.shape)
        # print(tokenized_prompts)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # x=x[:,0,:]
        # x = x @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PLOTPP.N_CTX
        ctx_init = cfg.TRAINER.PLOTPP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if False:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.PLOTPP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PLOTPP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class PromptLearner2(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)# the number of classes in the dataset
        n_ctx = cfg.TRAINER.PLOTPP.N_CTX#the number of context tokens 4
        n_ctx_vision = cfg.TRAINER.PLOTPP.N_CTX_V # the number of vision context tokens 2
        ctx_init_flag = cfg.TRAINER.PLOTPP.CTX_INIT #True
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0] #512
        clip_imsize = clip_model.visual.input_resolution #224
        cfg_imsize = cfg.INPUT.SIZE[0] #224
        M = cfg.TRAINER.PLOTPP.M #the number of our visual prompts 1
        N = cfg.TRAINER.PLOTPP.N   # the number of our text prompts 4
        dataset_name=cfg.DATASET['NAME']
        if dataset_name=='Caltech101':
            data_path='/ssd/lxin/PLOT/plot-pp/cache.json'
        elif dataset_name=='FGVCAircraft':
            data_path='/ssd/lxin/PLOT/aircraft_features3.json'
        elif dataset_name=="OxfordFlowers":
            data_path="/ssd/lxin/PLOT/plot-pp/flowers3_l.json"
        elif dataset_name=="StanfordCars":
            data_path='/ssd/lxin/PLOT/plot-pp/car.json'
        elif dataset_name=="DescribableTextures":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/dtd_dict.json"
        elif dataset_name=="EuroSAT":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/eurosat_dict.json"
        elif dataset_name=="Food101":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/food101_dict.json"
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/food_new_dict.json"             
        elif dataset_name=="OxfordPets":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/pets_new_dict.json"            
        elif dataset_name=="SUN397":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/sun397_dict.json"            
         
        elif dataset_name=="UCF101":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/ucf_dict.json"      
        
        elif dataset_name=="ImageNet":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/imagenet_dict.json"

        elif dataset_name=="ImageNetR":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/imagenet_r_dict.json"
        elif dataset_name=="ImageNetV2":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/imagenet_dict.json"
        elif dataset_name=="ImageNetA":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/imagenet_dict.json"                                                   
        elif dataset_name=="ImageNetSketch":
            data_path="/ssd/lxin/PLOT/plot-pp/descriptions/imagenet_dict.json"                                                   
            
            
        else:
            pass
        print(data_path)
            
            
        self.M = M
        self.N = N
        
            
      
        with open(data_path, 'r') as f:
       
            attribute_data = json.load(f)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        template_dict = {'Caltech101': ["a photo of a","this is a photo","this is picture of","one picture of a","one picture of a"], 
                         'DescribableTextures':['a photo of a texture', "this is a photo texture","this is a picture texture","one picture of a texture",'a photo of a texture', "this is a photo texture","this is a picture texture","one picture of a texture"],
                         'EuroSAT':['a centered satellite photo of', 'a centered satellite picture of','this is centered satellite photo of','one centered satellite photo of a','one centered satellite photo of a'], 
                         'FGVCAircraft':['a photo of an aircraft','a photo of an aircraft','a picture of an aircraft','this is aircraft picture of','one picture of an aircraft'],
                         'Food101':['a photo of a food', 'this is a food photo', ' this is food picture of','one picture of a food'], 
                         'ImageNet':["a photo of a","this is a photo ","this is a","one picture of a"],
                          'ImageNetR':["a photo of a","this is a photo ","this is a","one picture of a"],
                     'ImageNetA':["a photo of a","this is a photo ","this is a","one picture of a"], 
        'ImageNetSketch':["a photo of a","this is a photo ","this is a","one picture of a"],                                               
                          'ImageNetV2':["a photo of a","this is a photo ","this is a","one picture of a"],
                         'OxfordFlowers':['a photo of a flower','a photo of a flower', 'one picture of a flower','this is flower picture of','one picture of a flower'],
                         'OxfordPets':['a photo of a pet', 'one picture of a pet','this is pet picture of','one picture of a pet'],
                         'StanfordCars':["a photo of a","this is a photo ","this is picture of","one picture of a","one picture of a","one picture of a"],
                         'SUN397':["a photo of a","this is a photo","this is picture of","one picture of a"],
                         'UCF101':['a photo of a person doing', 'this is a photo people doing', 'this is picture of people doing', 'one picture of a person doing'],}
        
        if ctx_init_flag: #True
            ctx_list = template_dict[cfg.DATASET.NAME] #len=4 predicted templates
            n_ctx = len(ctx_list[0].split()) # 5 tokens
            ctx_vectors_list = []
            prompt_prefix_list = []
            
            for i in range(N): ## the number of our text prompts 4
                ctx_init = ctx_list[i].replace("_", " ")
                prompt = clip.tokenize(ctx_init)#torch.Size([1, 77])
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)#torch.Size([1, 77, 512])
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]#torch.Size([5, 512])
                ctx_vectors_list.append(ctx_vectors)
                prompt_prefix = ctx_init
                prompt_prefix_list.append(prompt_prefix)# predicted templates
            ctx_vision_vectors = torch.empty(M, n_ctx_vision ,768, dtype=dtype)#torch.Size([M=1, 2, 768])
            nn.init.normal_(ctx_vision_vectors, std=0.02)
            ctx_vectors = torch.stack(ctx_vectors_list)#torch.Size([4, 5, 512]) 
            
        else:
            ctx_vectors = torch.empty(N, n_ctx, ctx_dim, dtype=dtype)
            ctx_vision_vectors = torch.empty(M, n_ctx_vision ,768, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(ctx_vision_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.ctx = nn.Parameter(ctx_vectors) # parameters of text prompt to be learned torch.Size([4, 5, 512])
        self.ctx_vision = nn.Parameter(ctx_vision_vectors) # parameters of vision prompt to be learned torch.Size([1, 2, 768])
        
        classnames = [name.replace("_", " ") for name in classnames]#len=47
        # classnames = [name.replace("-", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_save = [prompt_prefix + " " + name + "." for name in classnames]# predicted templates + classname
        
        prompt_list = []
        if ctx_init: #one picture of a texture
            # for name in classnames:
                
                
            
            for i in range(N):
                prompt_prefix = prompt_prefix_list[i]
                prompts=[]
                if  True:
                    for name in classnames:
                        # print(attribute_data[name])
                        if i < len(attribute_data[name.lower()]):
                            if (dataset_name=='UCF101') or (dataset_name=='OxfordPets'):
                                
                                tmp=prompt_prefix+' '+name+', '+attribute_data[name.lower()][i]
                            else:
                                tmp=prompt_prefix+' '+name+' with '+attribute_data[name.lower()][i]
                            
                      
                           
                         
                        else:
                            tmp=prompt_prefix+' '+name+'.'
                            
                        # tmp=name+', '+attribute_data[name]['description']
                        # tmp=prompt_prefix+' '+name
                        prompts.append(tmp)                    
                # if True:
                        
                    
                # prompts = [prompt_prefix + " " + name + "." for name in classnames] # 47 predicted templates + classname
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # 47x77 
                prompt_list.append(tokenized_prompts)
            tokenized_prompts = torch.cat(prompt_list) #torch.Size([188=47*4, 77]) predicted templates + classname tokenlized
        else:
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            tokenized_prompts = tokenized_prompts.repeat(N,1)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)#torch.Size([188, 77, 512])
        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_save])  # (n_cls, n_tkn)
        # tokenized_prompts = tokenized_prompts.repeat(N,1)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls #47
        self.n_ctx = n_ctx # 5
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        # if label is not None:
        #     prefix = prefix[label]
        #     suffix = suffix[label]
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)
        ctx = ctx.permute(1, 0, 2, 3) #  N 100 16 512
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        return prompts
    
    def forward(self):

        ctx = self.ctx #torch.Size([4, 5, 512])
        ctx_vision = self.ctx_vision #torch.Size([1, 2, 768])
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            
        prefix = self.token_prefix #torch.Size([188, 1, 512])
        suffix = self.token_suffix #torch.Size([188, 71, 512])
        prompts = self.construct_prompts(ctx, prefix, suffix)
        
        return prompts, ctx_vision  # pass here original, as for visual 768 is required






class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.N = cfg.TRAINER.PLOTPP.N
        self.n_cls = len(classnames) #47

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        # print(tokenized_prompts.shape)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)
        # text_features = text_features.mean(dim=0)
        # print(text_features.shape)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class PLOTPP(TrainerX):
    """
    It is based on PLOT.
    """
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PLOTPP.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        name_tp_update = cfg.TRAINER.PLOTPP.MODEL_UPD
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.PLOTPP.PREC == "fp32" or cfg.TRAINER.PLOTPP.PREC == "amp":#False
            # CLIP's default precision is fp16
            clip_model.float()
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        # select params to update
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
                
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
                else:
                    if name_tp_update == "vision" and name_tp_update not in name:
                        param.requires_grad_(False)
                if "mapper" in name:
                    param.requires_grad_(True)
     # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                
        print(f"Parameters to be updated: {enabled}")
        
        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        
        if cfg.TRAINER.PLOTPP.PRETRAIN_DIR:
            load_pretrained_weights(self.model, cfg.TRAINER.PLOTPP.PRETRAIN_DIR)
        
        device_count = torch.cuda.device_count()
        if cfg.DATASET.NAME == 'ImageNet' or cfg.DATASET.NAME=='SUN397':
            self.device = torch.device("cuda")#("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            # self.model = nn.DataParallel(self.model)
            self.model.text_encoder.to(device1)
            self.model.text_encoder=nn.DataParallel(self.model.text_encoder)
        elif device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.device = torch.device("cuda")
            self.model.to(self.device)
            self.model = nn.DataParallel(self.model)
        else:
            self.device = torch.device("cuda:0")
            self.model.to(self.device)
        # NOTE: we give whole model to the optimizer, but only prompt_learner will be optimized
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.PLOTPP.PREC == "amp" else None
        print(self.device)
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)#torch.Size([32, 3, 224, 224])
        
        prec = self.cfg.TRAINER.PLOTPP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)#torch.Size([32, 102])
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
            
        loss_summary = {"loss": loss.item(),
                         "acc": compute_accuracy(output, label)[0].item()}
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
        
        
        