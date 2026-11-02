#!/bin/bash
# 激活 conda 环境并运行脚本

# # 初始化 conda（取消注释）
# __conda_setup="$('/home/daxiao/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/home/daxiao/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/home/daxiao/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/home/daxiao/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup



# 进入项目目录并运行脚本
cd /home/ssd/Merge_latest/recording

/home/daxiao/miniconda3/envs/daxiao2/bin/python app.py