#!/bin/bash
#SBATCH --job-name=FL_simulation
#SBATCH --output=fl_output_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=3  # 1 for the server and 2 for the clients

# Start the server in the background
srun --ntasks=1 python server.py &> server_output.txt> server_errors.txt &

# Give the server some time to initialize (adjust as needed)
sleep 10

# Start the clients
srun --ntasks=1 python client.py &> client1_output.txt> client1_errors.txt &
srun --ntasks=1 python client.py &> client2_output.txt> client2_errors.txt &

# Wait for all processes to complete
wait
