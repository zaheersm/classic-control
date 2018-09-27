#!/bin/bash -e
for value in {0..14}
do
    screen -d -m -S $value bash -c "./launch_run.sh $value config_files/actor_critic.json; exec bash"
done


