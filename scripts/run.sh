#!/bin/bash

# cd ..
#to c
# custom config
DATA='data/path'
TRAINER=MAP  
CFG='config_path'  # config file
CTP=end  # class token position (end or middle)
M=1 #  # number of vision prompts
N=4  # number of text prompts
NCTX=4  # number of context tokens
NCTX_V=4  # number of vision context tokens
SHOTS=16
CSC=False  # class-specific context (False or True)
DATASET='dtd'  # dataset name
SEED=1

 


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
PRETRAIN_DIR=None
# rm -r ${DIR}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"
    
    CUDA_VISIBLE_DEVICES=0 python train_b2n.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/vit.yaml \
    --output-dir ${DIR} \
    --device cuda:0 \
    --topk 10 \
    TRAINER.MAP.N_CTX ${NCTX} \
    TRAINER.MAP.CSC ${CSC} \
    TRAINER.MAP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}\
    DATASET.SUBSAMPLE_CLASSES base \
    TRAINER.MAP.M ${M}\
    TRAINER.MAP.N ${N} \
    TRAINER.MAP.N_CTX_V ${NCTX_V} \
    TRAINER.MAP.CTX_INIT True\
    TRAINER.MAP.MODEL_UPD "joint" 
fi




DIR=test/OP_N${N}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
PRETRAIN_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
rm -r ${DIR}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"
    
    CUDA_VISIBLE_DEVICES=0 python train_b2n.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file ${CFG} \
    --output-dir ${DIR} \
    --device cuda:0 \
    --eval-only \
    --lossweight 1 \
    --model-dir ${PRETRAIN_DIR} \
    --topk 10 \
    TRAINER.MAP.N_CTX ${NCTX} \
    TRAINER.MAP.CSC ${CSC} \
    TRAINER.MAP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}\
    DATASET.SUBSAMPLE_CLASSES new \
    TRAINER.MAP.M ${M}\
    TRAINER.MAP.N ${N} \
    TRAINER.MAP.N_CTX_V ${NCTX_V} \
    TRAINER.MAP.CTX_INIT True\
    TRAINER.MAP.MODEL_UPD "joint" #  
fi
done
done
done
 

