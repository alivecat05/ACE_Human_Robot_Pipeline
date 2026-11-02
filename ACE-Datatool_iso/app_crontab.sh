#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/daxiao/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/daxiao/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/daxiao/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/daxiao/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

cd /home/daxiao/ssd/ACE-Datatool
python push_feishu.py
python app_web.py