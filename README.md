# JeanZay_llm_inference_benchmark

Notes Si travail sur jz le code utilise directement les datasets et models dans le DSDIR, l'etape 1 et 2 peuvent donc être sauté
etape 1: telechargement des datasets  dans le dossier benchmarks/datasets
-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json
-https://huggingface.co/datasets/zhyncs/sonnet
-telechargement des datasets dans le dossier benchmarks/models: 
-https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
-https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
-https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct

etape 2: 
ajuster les adresses dans les fichiers du dossier config: config_datasets_paths_map,model_type_directories_map, model_type_map.json

etape 3: 
edit A100env
