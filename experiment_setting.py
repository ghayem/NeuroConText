
df_name = "body" # you should set either you want to work with title, abstract, or body

difumo_dimension=512 # difumo dimension could be different dimensionalities (64, 128, 256, 512, and 1024)

llm_key="Mistral-7B-v0.1" # possible keys are {
                       # "gpt-neo-125m", 
                       # "gpt-neo-125m-finetuned", 
                       # "gpt-neo-1.3B"
                       # "Mistral-7B-v0.1",
                       # "scibert",}

flag_train_size_control = False
train_size_control = 2000 # {2k, 3k, 5k, 7.5K, 10k, 12k, 15k, 17k, 19k}
