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
from trainers.cross_attention import CrossAttentionTransformer
from trainers.group_vit import GroupingBlock


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
        self.token_embedding=clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)#torch.Size([408, 77, 512])
        x = x.permute(1, 0, 2)  # NLD -> LND torch.Size([77, 408, 512])
        x = self.transformer(x) #torch.Size([77, 408, 512])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) #torch.Size([188, 77, 512])
        # x=x@ self.text_projection
        
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text(self,text):#torch.Size([408, 77])
        x=self.token_embedding(text).type(self.dtype)
        
        x =  x+ self.positional_embedding.type(self.dtype)#torch.Size([408, 77, 512])
        x = x.permute(1, 0, 2)  # NLD -> LND torch.Size([77, 408, 512])
        x = self.transformer(x) #torch.Size([77, 408, 512])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) #torch.Size([188, 77, 512])
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

import json     

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)# the number of classes in the dataset
        n_ctx = cfg.TRAINER.MAP.N_CTX#the number of context tokens 4
        n_ctx_vision = cfg.TRAINER.MAP.N_CTX_V # the number of vision context tokens 2
        ctx_init_flag = cfg.TRAINER.MAP.CTX_INIT #True
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0] #512
        clip_imsize = clip_model.visual.input_resolution #224
        cfg_imsize = cfg.INPUT.SIZE[0] #224
        M = cfg.TRAINER.MAP.M #the number of our visual prompts 1
        N = cfg.TRAINER.MAP.N   # the number of our text prompts 4
     
        dataset_name=cfg.DATASET['NAME']
        if dataset_name=='Caltech101':
            data_path='./map/cache.json'
        elif dataset_name=='FGVCAircraft':
            data_path='./aircraft.json'
        elif dataset_name=="OxfordFlowers":
            data_path="./map/flowers.json"
 
        elif dataset_name=="StanfordCars":
            data_path='./map/car.json'
        elif dataset_name=="DescribableTextures":
            data_path="./map/descriptions/dtd_dict.json"
        elif dataset_name=="EuroSAT":
            data_path='./map/des/eurosat.json'
        elif dataset_name=="Food101":
            data_path="./map/descriptions/food_new_dict.json"             
        elif dataset_name=="OxfordPets":
            data_path="./map/descriptions/pets_new_dict.json"            
        elif dataset_name=="SUN397":
            data_path="./map/descriptions/sun397_dict.json"            
         
        elif dataset_name=="UCF101":
            data_path="./map/descriptions/ucf_dict.json"      
        
        elif dataset_name=="ImageNet":
            data_path="./map/descriptions/imagenet_dict.json"            
            
        else:
            pass

           
        self.M = M
        self.N = N
        
            
      
        with open(data_path, 'r') as f:
       
            attribute_data = json.load(f)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        template_dict = {'Caltech101': ["a photo of a","this is a photo","this is picture of","one picture of a","one picture of a"], 
                         'DescribableTextures':['a photo of a texture', "this is a photo texture","this is a picture texture","one picture of a texture",'a photo of a texture', "this is a photo texture","this is a picture texture","one picture of a texture"],
                         'EuroSAT':['a centered satellite photo of', 'a centered satellite picture of','this is centered satellite photo of','one centered satellite photo of a','one centered satellite photo of a','one centered satellite photo of a','one centered satellite photo of a'], 
                         'FGVCAircraft':['a photo of an aircraft','a photo of an aircraft','a picture of an aircraft','this is aircraft picture of','one picture of an aircraft','one picture of an aircraft','one picture of an aircraft'],
                         'Food101':['a photo of a food', 'this is a food photo', ' this is food picture of','one picture of a food','one picture of a food','one picture of a food'], 
                         'ImageNet':["a photo of a","this is a photo ","this is a","one picture of a","one picture of a","one picture of a"],
                          'ImageNetR':["a photo of a","this is a photo ","this is a","one picture of a","one picture of a","one picture of a"],
                     'ImageNetA':["a photo of a","this is a photo ","this is a","one picture of a","one picture of a","one picture of a"], 
        'ImageNetSketch':["a photo of a","this is a photo ","this is a","one picture of a","one picture of a","one picture of a"],                                               
                          'ImageNetV2':["a photo of a","this is a photo ","this is a","one picture of a","one picture of a","one picture of a","one picture of a","one picture of a"],
                         'OxfordFlowers':['a photo of a flower','a photo of a flower', 'one picture of a flower','this is flower picture of','one picture of a flower','a photo of a flower','a photo of a flower',],
                         'OxfordPets':['a photo of a pet', 'one picture of a pet','this is pet picture of','one picture of a pet','one picture of a pet','one picture of a pet'],
                         'StanfordCars':["a photo of a","this is a photo ","this is picture of","one picture of a","one picture of a","one picture of a"],
                         'SUN397':["a photo of a","this is a photo","this is picture of","one picture of a","one picture of a","one picture of a"],
                         'UCF101':['a photo of a person doing', 'this is a photo people doing', 'this is picture of people doing', 'one picture of a person doing','one picture of a person doing','one picture of a person doing'],}
        
    
        
        if ctx_init_flag: #True
            ctx_list = template_dict['ImageNet'] #len=4 predicted templates#cfg.DATASET.NAME
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
        prompts = [prompt_prefix + " " + name + "." for name in classnames]# predicted templates + classname
        
        prompt_list = []
        if ctx_init: #one picture of a texture

            for i in range(N):
                prompt_prefix = prompt_prefix_list[i]
                prompts=[]
                if  True:
                    for name in classnames:
                       
                        if i < len(attribute_data[name.lower()]):
                            if (dataset_name=='UCF101') or (dataset_name=='OxfordPets'):
                                
                                tmp=prompt_prefix+' '+name+', '+attribute_data[name.lower()][i]
                            else:
                                tmp=prompt_prefix+' '+name+' with '+attribute_data[name.lower()][i]             
                        else:
                            tmp=prompt_prefix+' '+name+'.'
                
                        prompts.append(tmp)                    
         
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # 47x77 
                prompt_list.append(tokenized_prompts)
            tokenized_prompts = torch.cat(prompt_list) #torch.Size([188=47*4, 77]) predicted templates + classname tokenlized
        else:
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            tokenized_prompts = tokenized_prompts.repeat(N,1)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)#torch.Size([188, 77, 512])
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls #47
        self.n_ctx = n_ctx # 5
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
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

        ctx = self.ctx 
        ctx_vision = self.ctx_vision 
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            
        prefix = self.token_prefix 
        suffix = self.token_suffix 
        prompts = self.construct_prompts(ctx, prefix, suffix)
        
        return prompts, ctx_vision   
class MLP_mapper(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.mapper1=nn.Linear(512,dim)
        self.act=nn.ReLU()
        self.mapper2=nn.Linear(dim,768)
    def forward(self,x):
        
       
        return self.mapper2(self.act(self.mapper1(x.float())))        
        
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.loss_weight=cfg.loss_weight
        self.topk=cfg.topk
        self.layer=cfg.layer
    
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts #torch.Size([188=4*47, 77])
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.N = cfg.TRAINER.MAP.N #4
        self.n_cls = len(classnames) #47
        self.eps = 0.1
        self.max_iter = 100
        self.dataset =  cfg.DATASET.NAME #'DescribableTextures'
        self.dataset_flag=cfg.DATASET.FLAG
        
        
        self.n_ctx_vision = cfg.TRAINER.MAP.N_CTX_V
        self.mapper=MLP_mapper(cfg.TRAINER.MAP.MAPPER_DIM) 
        self.group_mapper=GroupingBlock(dim=cfg.GDIM1,out_dim=cfg.GDIM2,num_heads=cfg.NHEAD,num_group_token=self.n_ctx_vision,num_output_group=self.n_ctx_vision,norm_layer=nn.LayerNorm).half()
        self.attention_mapper=CrossAttentionTransformer(num_layers=cfg.CNLAYER,d1=cfg.CDIM1,d2=cfg.CDIM2,num_heads=cfg.NHEAD)
        if self.dataset== 'ImageNet' or self.dataset=='SUN397':
            self.device = torch.device('cuda:0')
            self.device1 = torch.device("cuda")
        else:
            self.device = torch.device(cfg['DEVICE'])
            
            self.device1 = torch.device("cuda")

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    
    def acc_plot(self,image_features,text_features,M,N=4,b=64):
        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()
        sim = sim.view(M,self.N,b*self.n_cls)
        sim = sim.permute(2,0,1)
        wdist = 1.0 - sim
        xx=torch.zeros(b*self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy=torch.zeros(b*self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK,xx,yy)
        if torch.isnan(T).any():
            image_features_pool=image_features.mean(dim=0)
            image_features_pool=F.normalize(image_features_pool,dim=1)
            text_features_pool=text_features.mean(dim=0)
            text_features_pool=F.normalize(text_features_pool,dim=1)
            logits=self.logit_scale.exp()*image_features_pool@text_features_pool.t()
            return logits
            return None
        
        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,self.n_cls)#torch.Size([32, 102])
        logit_scale = self.logit_scale.exp()
        logits2 = logit_scale * sim_op
        return logits2




            
    def cross_attn2(self,vis_ctx,text_features):#torch.Size([16, 4, 768]),torch.Size([4, 102, 768])

        text_features_v=text_features.reshape(text_features.shape[0],-1,text_features.shape[-1])
        vis_ctx=self.attention_mapper(text_features_v,vis_ctx.float())

        return vis_ctx 


    def image_encode(self,visiontransformer,x,vision_prompt,text_features):#v_p：torch.Size([1, 4, 64, 768]
        bs = x.shape[0]
        x = visiontransformer.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([visiontransformer.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visiontransformer.positional_embedding.to(x.dtype)#torch.Size([32, 197, 768])
        N_t=vision_prompt.shape[1]
    
        if vision_prompt.dim() >= 3:
            x = x.unsqueeze(0).expand(vision_prompt.shape[0],-1,-1,-1)#torch.Size([1, 32, 197, 768])
            visual_ctx = vision_prompt.unsqueeze(1).expand(-1,x.shape[1],-1,-1) #torch.Size([1, 32, 2, 768])
            x = torch.cat([x, visual_ctx], dim=-2)#torch.Size([1, 32, 199, 768])
            x = x.contiguous().view(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])#torch.Size([32, 199, 768])
            x = visiontransformer.ln_pre(x)#torch.Size([32, 199, 768])
            x = x.permute(1, 0, 2)  # NLD -> LND torch.Size([199, 32, 768])
            for i in range(len(visiontransformer.transformer.resblocks)):
                b=visiontransformer.transformer.resblocks[i]
                x=b(x)
                if i==self.layer:
                    x_t=x.permute(1,0,2)#torch.Size([32, 201, 768])
                    vis_ctx=x_t[:,-N_t:,:]#torch.Size([16, 4, 768])
                    visual_tokens=x_t[:,1:-N_t,:]
                    x_cls_token=x[0] #torch.Size([16, 768])
                    text_features_mean=torch.mean(text_features,dim=0)# [102, 768]
                    top_k=int(len(text_features_mean)/4)#25
                    if len(text_features_mean)<self.topk:
                        top_k=len(text_features_mean)
                    else:
                        top_k=self.topk #10
               
                    cosine_similarity=F.cosine_similarity(x_cls_token.unsqueeze(1),text_features_mean,dim=2)#torch.Size([16, 102])
                    topk_similar_indices = cosine_similarity.topk(top_k, dim=1).indices.squeeze()#torch.Size([16, 25])
                    text_features_v=text_features.permute(1,0,2)#torch.Size([102, 4, 768])
                    text_features_v_flattened=text_features_v.reshape(text_features_v.shape[0],-1)#torch.Size([102, 3072])
                    
                    selected_features=text_features_v_flattened[topk_similar_indices] #torch.Size([16, 25, 3072])
                    selected_features=selected_features.reshape(selected_features.shape[0],top_k,text_features_v.shape[1],text_features_v.shape[2])
                    
                
                    vis_ctx=self.cross_attn2(vis_ctx,selected_features)#torch.Size([32, 4, 768])
                    x[-N_t:,:,:]=vis_ctx.permute(1,0,2)


                if i==self.layer+1 and not self.dataset_flag:
                    x_t=x.permute(1,0,2)#torch.Size([32, 201, 768])
                    vis_ctx=x_t[:,-N_t:,:]#torch.Size([16, 4, 768])
                    visual_tokens=x_t[:,1:-N_t,:]
                    vis_ctx2,_=self.group_mapper(visual_tokens,vis_ctx)#torch.Size([32, 4, 768])
                    # # vis_ctx=vis_ctx+vis_ctx_tmp torch.cat([x_t[:,:-N_t,:],])
                    # vis_ctx=self.cross_attn(vis_ctx,text_features)
                    x=torch.cat([x[:-N_t,:,:],vis_ctx2.permute(1,0,2)],dim=0)
                    # x[-N_t:,:,:]=vis_ctx2.permute(1,0,2)  
                    
            # x = visiontransformer.transformer(x)#torch.Size([199, 32, 768])
            x = x.permute(1, 0, 2)  # LND -> NLD #torch.Size([32, 199, 768])

            x = x.contiguous().view(vision_prompt.shape[0], bs, x.shape[1], x.shape[-1]) #torch.Size([1, 32, 199, 768])
            visual_ctx=visiontransformer.ln_post(x[:,:,-N_t:,:])
            x = visiontransformer.ln_post(x[:,:, 0, :])# torch.Size([1, 32, 768]) cls token torch.Size([1, 64, 768])
          
        else: 
            if visiontransformer.VPT_shallow:
                visual_ctx = vision_prompt.expand(x.shape[0], -1, -1).half()
                x = torch.cat([x, visual_ctx], dim=1)
            else:
                assert visiontransformer.vision_prompt_till_layer_visual == 0

            # Normal code as before
            x = visiontransformer.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = visiontransformer.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            
            x = visiontransformer.ln_post(x[:, 0, :])
            
        if visiontransformer.proj is not None:
            x = x @ visiontransformer.proj #torch.Size([1, 32, 512])
            visual_ctx=visual_ctx@visiontransformer.proj
            
          

        return x,visual_ctx     
    

    

    def forward(self, image,dis=False):
        
        b = image.shape[0] #32
        prompts, vision_prompts = self.prompt_learner()# prompts learnable // prompts construct  torch.Size([1, 4, 768])
        tokenized_prompts = self.tokenized_prompts #initialized by templates
   
        image_features,v_p = self.image_encoder(image.type(self.dtype),vision_prompts) #torch.Size([1, 32, 512])
        image_feature_pool = image_features.mean(dim=0) #torch.Size([32, 512])vision_prompts
        M = image_features.shape[0]#1
        self.d = image_features.shape[-1]#512
        
        if self.dataset == 'ImageNet' or self.dataset=='SUN397' :
            text_features = self.text_encoder(prompts.to(self.device1), tokenized_prompts.to(self.device1)) 
            text_features = text_features.to(self.device1)
            tokenized_prompts=tokenized_prompts.to(self.device1)
            text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
            # text_features_v=self.text_encoder.module.encode_text(tokenized_prompts).contiguous().view(self.N, self.n_cls, self.d)
            
            text_feature_pool = text_features.mean(dim=0)
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts).contiguous().view(self.N, self.n_cls, self.d)#torch.Size([4, 102, 512])
            tokenized_prompts=tokenized_prompts.to(self.device)
        
            text_feature_pool = text_features.mean(dim=0)#torch.Size([102, 512])  

        new_text_features=self.mapper(text_features)#torch.Size([4, 102, 768])
   
   
        image_features,v_p = self.image_encode(self.image_encoder,image.type(self.dtype),vision_prompts,new_text_features) #torch.Size([1, 32, 512])
  
        
        v_p=v_p[0].permute(1,0,2)

        logits_4=self.acc_plot(v_p,text_features,v_p.shape[0],b=b)

        image_feature_pool = image_features.mean(dim=0) #torch.Size([32, 512])
        image_features =  F.normalize(image_features, dim=2)  # N c d  torch.Size([1, 32, 512])
        image_feature_pool = F.normalize(image_feature_pool, dim=1) #torch.Size([32, 512])
        text_features = F.normalize(text_features, dim=2) #torch.Size([4, 47, 512])
        text_feature_pool = F.normalize(text_feature_pool, dim=1) #torch.Size([47, 512])

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous() #torch.Size([1, 32, 512]) ，torch.Size([4, 102, 512]) torch.Size([1, 4, 32, 102])
        sim = sim.view(M,self.N,b*self.n_cls)#torch.Size([1, 4, 3264])
        sim = sim.permute(2,0,1)#torch.Size([3264, 1, 4])
        wdist = 1.0 - sim #torch.Size([3264, 1, 4])
        xx=torch.zeros(b*self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        yy=torch.zeros(b*self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
        
        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK,xx,yy)
        
        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,self.n_cls)#torch.Size([32, 102])      
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feature_pool @ text_feature_pool.t()#torch.Size([32, 512]) torch.Size([102, 512])
        logits2 = logit_scale * sim_op
        return logits2 +logits_4*self.loss_weight 

@TRAINER_REGISTRY.register()
class MAP(TrainerX):
 
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAP.PREC in ["fp16", "fp32", "amp"]
    
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        name_tp_update = cfg.TRAINER.MAP.MODEL_UPD
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MAP.PREC == "fp32" or cfg.TRAINER.MAP.PREC == "amp":#False
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
                    if not self.cfg.DATASET.TACT:
                        name_tp_update="vision"
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
        

        if cfg.TRAINER.MAP.PRETRAIN_DIR:
            load_pretrained_weights(self.model, cfg.TRAINER.MAP.PRETRAIN_DIR)
        
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
        
        self.scaler = GradScaler() if cfg.TRAINER.MAP.PREC == "amp" else None
        print(self.device)
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)#torch.Size([32, 3, 224, 224])
        
        prec = self.cfg.TRAINER.MAP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output= self.model(image,True)#torch.Size([32, 102])
            loss = F.cross_entropy(output, label)#+sum*0.1
          
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

    
        model_file = "model-best.pth.tar"
        epoch=20

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
        
        
        
        
        