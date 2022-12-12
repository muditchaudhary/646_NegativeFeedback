# Fine Tuning Query Represetation using Iterative Negative Feedback
by Mudit Chaudhary, Dhawal Gupta

## Datasets
We provide pre-processed version of training and dev sets from MS-Marco here: https://drive.google.com/drive/folders/1SWb4w8TrLmHkEgGM8GOCdNHXn2EdF7iN?usp=sharing

## Generating and storing cached representations
```bash
python src/rankers/dpr.py --input_file <input file path> \
 --query_cache <query representation save path> \
 --passage_cache <passage representation save path>
```

## Training and Evaluation
```bash
python src/launch.py --alpha1=4 --alpha2=10 --alpha3=4 \
--cached_embeddings_root=./data/cached_embeddings/ \ # Cached embedding root folder
--dataset_folder=./data/processed_data/ \ # Dataset root folder
--embedding_type=normalized_embedding \ 
--epochs=20 \ 
--learning_rate=0.00172854898644987 \
--max_refining_iterations=5 \
--neg_sample_rank_from=900 \
--neg_sample_rank_to=1000 \
--neg_sampling_ranker=dpr \
--num_neg_samples=20 \
--partial_eval_steps=100 \ # To perform partial evaluation on dev set. Set to None for full eval
--save_model_root=./saved_models \
--train_batch_size=3 \
--warmup_percent=0.2968326853108489 \
--use_wandb=True \ # For wandb logging \
--save_preds_root ./saved_preds/ \
--eval_only # For performing only evaluation and no training
```
