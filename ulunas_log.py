import subprocess

# 启动A方案的TensorBoard服务
log_dir = 'output_ulunas/logs'
cmd = ['tensorboard', f'--logdir={log_dir}', '--port=6006']
print(f"启动output_ulunas方案TensorBoard: {cmd}")
subprocess.run(cmd, check=True)