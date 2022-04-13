# gradient-norm-penalty
This is a temperory repo for ICML. 

The config file is in the file folder. The main config file must be specified, which is the path to the train_config.py file. All the params could be changed in the files at config folder. Also, the training config could be changed via parsing flags when executing. 


An exmaple:

<code>
python3 -m gnp.main.main  --config=${YOUR_TRAIN_CONFIG_DIR}
                          --working_dir=${YOUR_OURPUT_DIR} 
                          --config.gnp.alpha=$alpha 
                          --config.gnp.r=$ra 
                          --config.opt.opt_type=$opt_type 
</code>

\
One could also add costom models, optimizers in the corerspondin folder, and register them in the get_* function. If you encounter any trouble or would like to discuss, it is welcomed to leave a message.
