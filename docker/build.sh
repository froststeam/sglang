#!/bin/bash
# run: nohup bash build.sh > build.log 2>&1 &
# stop: pkill -f build.sh
# check log: tail -f build.log
set -ex
eval "$(ssh-agent -s)"
ssh-add $HOME/.ssh/id_ed25519
while true; do
    git fetch origin
    git checkout musa/v0.5.10.post1
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse @{u})
    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "New commit found, building new image..."
        git pull
        docker build --push --build-arg CACHE_DATE="$(date)" --ssh default --build-arg GPU_ARCH=mp_31 \
            -t sh-harbor.mthreads.com/mcc-ai-serving/sglang:v0.5.10.post1-py310-4.3.5 -f Dockerfile.musa .
        docker tag sh-harbor.mthreads.com/mcc-ai-serving/sglang:v0.5.10.post1-py310-4.3.5 \
            registry.mthreads.com/mcconline/inference/sglang:v0.5.10.post1-py310-4.3.5
        docker push registry.mthreads.com/mcconline/inference/sglang:v0.5.10.post1-py310-4.3.5
    fi
    echo "Current date: $(date)"
    sleep 3600
done
