nvidia-smi

# nohup python -u train.py --batch_size 64 --max_len 200 --epochs 1 --embed_dim 3 --lr 0.02 > nohup/nohup1.out &

# # 使用 tensorboard
# nohup python -u train.py --batch_size 64 --max_len 200 --epochs 1 --embed_dim 3 --lr 0.02 --use_tensorboard > nohup/nohup2.out &
# tensorboard --logdir="./tensorboards" --bind_all --port=6006 --purge_orphaned_data=true
# # 在浏览器中打开 http://localhost:6006/ 查看
# # 如果是服务器：http://10.192.9.235:20038/，具体地址需要查看服务器的说明


# 使用 multi-gpu
nohup python -u train.py --multi_gpu --batch_size 64 --max_len 200 --epochs 10 --embed_dim 3 --lr 0.02 > nohup/nohup3.out &

ps -aux | grep train