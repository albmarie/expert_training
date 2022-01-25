import argparse
import os
import getpass

####################################################################################################
####################################################################################################
####################################################################################################

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", help="<REQUIRED> Path to folder containing datasets", type=str, required=True)
	parser.add_argument("-w", "--weights", help="<REQUIRED> Path to folder where weights will be saved", type=str, required=True)
	parser.add_argument("-c", "--csv", help="<REQUIRED> Path to folder where csv files will be saved", type=str, required=True)
	parser.add_argument("-i", "--init", help="Weights to use at initialisation (by default, it is a pre-trained model on losslessly compressed images).", type=str, required=False)
	parser.add_argument('--gui', help="Allow GUI inside docker container (by default, False). It may be needed for specific scripts.", dest='gui', action='store_true')
	parser.add_argument('--gpu', help="Allow GPU utilisation (by default, False)", dest='gpu', action='store_true')
	parser.add_argument('--singularity', help="Run a command using singularity instead of docker (by default, False)", dest='singularity', action='store_true')
	parser.add_argument('--no_cache_link', help="Link cache folder used by pytorch to not download model weights at every execution (by default, True)", dest='no_cache_link', action='store_true')
	opt = parser.parse_args()

	script_folder = os.path.dirname(os.path.abspath(__file__))
	if not opt.singularity:
		os.system("sudo docker build -t expert_training_container:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) $(dirname ${0})")
		print("#" * 50)
	
	cmd = "xhost + && " if opt.gui else ""
	user = getpass.getuser()
	if opt.singularity:
		cmd += "singularity run "
		cmd += "--nv " if opt.gpu else ""
		cmd += "--bind /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY " if opt.gui else ""
		if opt.init is not None:
			cmd += "--bind " + os.path.dirname(opt.init) + "/:/init/ "
		if not opt.no_cache_link:
			cmd += "--bind /home/" + user + "/.cache/torch/hub/checkpoints/:/home/user/.cache/torch/hub/checkpoints/ "
		cmd += "--bind " + os.path.abspath(opt.data) + ":/data/ "
		cmd += "--bind " + os.path.abspath(opt.weights) + ":/weights/ "
		cmd += "--bind " + os.path.abspath(opt.csv) + ":/csv/ "
		cmd += "--bind " + script_folder + "/python/:/python/ "
		cmd += script_folder + "/singularity/expert_training_container.sif /bin/bash -c "
		cmd += "\"cd /python/ && python3 expert_training.py"
		if opt.init is not None:
			cmd += " --init /init/" + os.path.basename(opt.init)
		cmd += "\""
	else:
		cmd += "sudo docker run -it "
		cmd += "--shm-size=32g --gpus all --rm " if opt.gpu else ""
		cmd += "-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY " if opt.gui else ""
		cmd += "--mount type=tmpfs,destination=/mnt/ram_partition,tmpfs-size=512m "
		if opt.init is not None:
			cmd += "-v " + os.path.dirname(opt.init) + "/:/init/ "
		if not opt.no_cache_link:
			cmd += "-v /home/" + user + "/.cache/torch/hub/checkpoints/:/home/user/.cache/torch/hub/checkpoints/ "
		cmd += "-v " + os.path.abspath(opt.data) + ":/data/ "
		cmd += "-v " + os.path.abspath(opt.weights) + ":/weights/ "
		cmd += "-v " + os.path.abspath(opt.csv) + ":/csv/ "
		cmd += "-v " + script_folder + "/python/:/python/ "
		cmd += "expert_training_container:latest /bin/bash -c "
		cmd += "\"cd /python/ && python3 expert_training.py"
		if opt.init is not None:
			cmd += " --init /init/" + os.path.basename(opt.init)
		cmd += "\""

	#print("cmd", cmd)
	os.system(cmd)

####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == "__main__":
    main()