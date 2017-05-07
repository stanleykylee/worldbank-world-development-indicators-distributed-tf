#!/bin/bash

# declare PS and WORKER nodes
AWS_PS_NODES=( 34.201.134.175 34.201.41.28 )
AWS_WORKER_NODES=( 52.90.51.253 174.129.102.87 )

# create and send run script for PS nodes
TASK_INDEX=0
for AWS_NODE in ${AWS_PS_NODES[@]}; do
    echo "pushing to ps${TASK_INDEX} - ${AWS_NODE}"
    echo -e "ssh -i \"cs498-dist-tf.pem\" ubuntu@${AWS_NODE}" > "connect_ps${TASK_INDEX}.sh"
    echo -e "python trainer.py \\
    --ps_hosts=${AWS_PS_NODES[0]}:2222,${AWS_PS_NODES[1]}:2222 \\
    --worker_hosts=${AWS_WORKER_NODES[0]}:2222,${AWS_WORKER_NODES[1]}:2222 \\
    --job_name=ps --task_index=${TASK_INDEX}" > run.sh
    chmod +x run.sh "connect_ps${TASK_INDEX}.sh"
    ((TASK_INDEX++))
    scp -i "cs498-dist-tf.pem" run.sh "ubuntu@${AWS_NODE}:/home/ubuntu/tf/src/run.sh"
    scp -i "cs498-dist-tf.pem" trainer.py "ubuntu@${AWS_NODE}:/home/ubuntu/tf/src/trainer.py"
done

# create and send run script for WORKER nodes
TASK_INDEX=0
for AWS_NODE in ${AWS_WORKER_NODES[@]}; do
    echo "pushing to worker${TASK_INDEX} - ${AWS_NODE}"
    echo -e "ssh -i \"cs498-dist-tf.pem\" ubuntu@${AWS_NODE}" > "connect_worker${TASK_INDEX}.sh"
    echo -e "python trainer.py \\
    --ps_hosts=${AWS_PS_NODES[0]}:2222,${AWS_PS_NODES[1]}:2222 \\
    --worker_hosts=${AWS_WORKER_NODES[0]}:2222,${AWS_WORKER_NODES[1]}:2222 \\
    --job_name=worker --task_index=${TASK_INDEX}" > run.sh
    chmod +x run.sh "connect_ps${TASK_INDEX}.sh"
    ((TASK_INDEX++))
    scp -i "cs498-dist-tf.pem" run.sh "ubuntu@${AWS_NODE}:/home/ubuntu/tf/src/run.sh"
    scp -i "cs498-dist-tf.pem" trainer.py "ubuntu@${AWS_NODE}:/home/ubuntu/tf/src/trainer.py"
done
exit 0