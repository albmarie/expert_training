import argparse
import os

####################################################################################################
####################################################################################################
####################################################################################################

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", help="<REQUIRED> Path to folder containing datasets (the base one)", type=str, required=True)
	parser.add_argument('--gui', help="Allow GUI inside docker container (by default, False). It may be needed for specific scripts.", dest='gui', action='store_true')
	parser.add_argument('--singularity', help="Run a command using singularity instead of docker (by default, False)", dest='singularity', action='store_true')
	opt = parser.parse_args()

	script_folder = os.path.dirname(os.path.abspath(__file__))
	if not opt.singularity:
		os.system("sudo docker build -t expert_training_container:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) $(dirname ${0})")
		print("#" * 50)

	cmd = "xhost + && " if opt.gui else ""
	if opt.singularity:
		cmd += "singularity run "
		cmd += "--bind /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY " if opt.gui else ""
		cmd += "--bind " + os.path.abspath(opt.data) + ":/data/ "
		cmd += "--bind " + script_folder + "/python/:/python/ "
		cmd += script_folder + "/singularity/expert_training_container.sif /bin/bash -c "
		cmd += "\"cd /python/ && python3 dataset_preprocessing.py\""
	else:
		cmd += "sudo docker run -it "
		cmd += "-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY " if opt.gui else ""
		cmd += "--mount type=tmpfs,destination=/mnt/ram_partition,tmpfs-size=512m "
		cmd += "-v " + os.path.abspath(opt.data) + ":/data/ "
		cmd += "-v " + script_folder + "/python/:/python/ "
		cmd += "expert_training_container:latest /bin/bash -c "
		cmd += "\"cd /python/ && python3 dataset_preprocessing.py\""
	os.system(cmd)

####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == "__main__":
    main()