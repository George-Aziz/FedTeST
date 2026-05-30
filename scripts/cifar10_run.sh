cd ..

#FedTeST Progressive Dirichlet
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=progressive --activate_refine_round=300 --adapt_target=attn_out
; #FedTeST Batch Dirichlet
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=batch --activate_refine_round=1 --adapt_target=attn_out
; #FedTeST Online Dirichlet
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=online_avg --activate_refine_round=1 --adapt_target=attn_out
; #FedTP Dirichlet
python main.py --model=vit --dataset=cifar10 --alg=FedTP --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
; #FedTeST Progressive Label Uni (Pathological)
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=progressive --activate_refine_round=300 --adapt_target=attn_out
; #FedTeST Batch Label Uni (Pathological)
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=batch --activate_refine_round=1 --adapt_target=attn_out
; #FedTeST Online Label Uni (Pathological)
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=online_avg --activate_refine_round=300 --adapt_target=attn_out
; #FedTP Label Uni (Pathological)
python main.py --model=vit --dataset=cifar10 --alg=FedTP --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag
;
#FedTeST Progressive Optimal Adaptive Module Refinement Activation Round Check with Dirichlet (Excluding 300 as that is in the section above)
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=progressive --activate_refine_round=1 --adapt_target=attn_out
; 
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=progressive --activate_refine_round=100 --adapt_target=attn_out
; 
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=progressive --activate_refine_round=200 --adapt_target=attn_out
; 
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=progressive --activate_refine_round=400 --adapt_target=attn_out

#FedTeST Batch & Online Optimal Adaptive Module Refinement Activation Round Check with Label Uni (Pathological) 
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=batch --activate_refine_round=1 --adapt_target=attn_out
; 
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=online_avg --activate_refine_round=1 --adapt_target=attn_out
; 
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=online_avg --activate_refine_round=400 --adapt_target=attn_out
;
#Naive Progressive Dirichlet
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=progressive --activate_refine_round=300 --adapt_target=all
; #Naive Batch Dirichlet
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=batch --activate_refine_round=1 --adapt_target=all
; #Naive Online Dirichlet
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=online_avg --activate_refine_round=1 --adapt_target=all
; #Naive Progressive Label Uni (Pathological)
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=progressive --activate_refine_round=300 --adapt_target=all
; #Naive Batch Label Uni (Pathological)
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=batch --activate_refine_round=1 --adapt_target=all
; #Naive Online Label Uni (Pathological)
python main.py --model=vit --dataset=cifar10 --alg=FedTeST --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag --train_adapt_method=online_avg --activate_refine_round=1 --adapt_target=all
;
#Other Baselines (FedAvg / FedProx / FedPer / pFedMe / FedBN / pFedHN / FedRoD / Personalized-T) with Dirichlet
python main.py --model=cnn --dataset=cifar10 --alg=fedavg --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=fedprox --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=fedPer --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=pfedMe --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn-b --dataset=cifar10 --alg=fedBN --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=pFedHN --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=fedRod --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --balanced_soft_max --wandb_flag 
;
python main.py --model=vit --dataset=cifar10 --alg=fedavg --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=vit --dataset=cifar10 --alg=Personalized-T --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeldir --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag
;
#Other Baselines (FedAvg / FedProx / FedPer / pFedMe / FedBN / pFedHN / FedRoD / Personalized-T) with Label Uni (Pathological)
python main.py --model=cnn --dataset=cifar10 --alg=fedavg --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=fedprox --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=fedPer --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=pfedMe --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn-b --dataset=cifar10 --alg=fedBN --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=pFedHN --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=cnn --dataset=cifar10 --alg=fedRod --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --balanced_soft_max --wandb_flag
;
python main.py --model=vit --dataset=cifar10 --alg=fedavg --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 
;
python main.py --model=vit --dataset=cifar10 --alg=Personalized-T --lr=0.01 --batch-size=64 --epochs=5 --n_parties=50 --mu=0.01 --rho=0.9 --comm_round=500 --partition=noniid-labeluni --beta=0.3 --device='cuda:0' --datadir='./data/' --logdir='./logs/' --noise=0 --sample=0.1 --init_seed=0 --train_acc_pre --test_round=400 --wandb_flag 