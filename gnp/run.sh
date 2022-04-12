export CUDA_VISIBLE_DEVICES=0,1,2,3
source /home/zy/work/virtual_env/jax/bin/activate

SOURCE_PATH=/home/zy/work/code/module_pkg/gradient_norm_penalty_new_jax/
cd $SOURCE_PATH

TRAIN_CONFIG_DIR=/home/zy/work/code/module_pkg/gradient_norm_penalty_new_jax/gnp/config/train_config.py
WORKING_DIR=/home/zy/models/gnp_convert/new_jax_second_edition

declare -a SEED=(6666)
declare -a ALPHA=(0.8)
declare -a R=(0.1)
declare -a BLA=(cutout)
declare -a DS=(cifar10)
declare -a OPT=(Momentum)


for alpha in ${ALPHA[@]}; do
    for seeds in ${SEED[@]}; do
        for ra in ${R[@]}; do
            for bla in ${BLA[@]}; do
                for ds in ${DS[@]}; do
                    for opt_type in ${OPT[@]}; do
                        CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m gnp.main.main  --config=${TRAIN_CONFIG_DIR} \
                                                                                --working_dir=${WORKING_DIR} \
                                                                                --config.seeds=$seeds \
                                                                                --config.gnp.alpha=$alpha \
                                                                                --config.gnp.r=$ra \
                                                                                --config.dataset.batch_level_augmentations=$bla \
                                                                                --config.dataset.dataset_name=$ds \
                                                                                --config.opt.opt_type=$opt_type \
                        sleep 5s
                    done
                done
            done
        done
    done
done