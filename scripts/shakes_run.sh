# Dataset Setup
cd ../data/shakespeare
python generate_data.py --s_frac 0.7 --tr_frac 0.8 --seed 12345

# Experiments
cd ../..
# FedTeST Progressive 
python main.py --model=transformer --dataset=shakespeare --alg=FedTeST  --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag --train_adapt_method=progressive --activate_refine_round=150 --adapt_target=attn_out
; # FedTeST Batch 
python main.py --model=transformer --dataset=shakespeare --alg=FedTeST  --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag --train_adapt_method=batch --activate_refine_round=1 --adapt_target=attn_out
; # FedTeST Online 
python main.py --model=transformer --dataset=shakespeare --alg=FedTeST  --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag --train_adapt_method=online_avg --activate_refine_round=1 --adapt_target=attn_out
# Naive Progressive 
python main.py --model=transformer --dataset=shakespeare --alg=FedTeST  --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag --train_adapt_method=progressive --activate_refine_round=150 --adapt_target=all
; # Naive Batch 
python main.py --model=transformer --dataset=shakespeare --alg=FedTeST  --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag --train_adapt_method=batch --activate_refine_round=1 --adapt_target=all
; # Naive Online 
python main.py --model=transformer --dataset=shakespeare --alg=FedTeST  --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag --train_adapt_method=online_avg --activate_refine_round=1 --adapt_target=all
; # FedTP
python main.py --model=transformer --dataset=shakespeare --alg=FedTP --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag 
; # FedAvg
python main.py --model=lstm --dataset=shakespeare --alg=fedavg --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag 
; # FedProx
python main.py --model=lstm --dataset=shakespeare --alg=fedprox --lr=0.01 --batch-size=64 --epochs=1 --mu=0.01 --comm_round=300  --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag 
; # FedPer
python main.py --model=lstm --dataset=shakespeare --alg=fedPer --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag 
; # pFedMe
python main.py --model=lstm --dataset=shakespeare --alg=pfedMe --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag 
; # FedAvg-T
python main.py --model=transformer --dataset=shakespeare --alg=fedavg --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag 
; #Personalized-T
python main.py --model=transformer --dataset=shakespeare --alg=Personalized-T --lr=0.01 --batch-size=64 --epochs=1 --comm_round=300 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --sample=0.1 --depth=2 --chunk_len=10 --init_seed=0 --test_round=200 --wandb_flag