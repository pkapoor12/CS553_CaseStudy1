#! /bin/bash

PORT=22014
MACHINE=paffenroth-23.dyn.wpi.edu
STUDENT_ADMIN_KEY_PATH=$HOME/.ssh

COMMAND="ssh -i ~/.ssh/secure_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE}"

# Check if ssh connection can be established with current secure key (or if secure key exists at all)
if ! eval "$COMMAND 'exit'"; then
	# SSH connection failed with secure key (or secure key doesn't exist), create new secure key and copy it to server
	echo "Failed to establish connection with secure key; creating new key..."

	# For testing with already existing secure key: script shouldn't be entering this conditional block
	#exit

	# Clean up
	ssh-keygen -f ~/.ssh/known_hosts -R "[paffenroth-23.dyn.wpi.edu]:22014"
	rm -rf tmp

	# Make new tmp dir
	mkdir tmp

	# Copy key to tmp
	cp ${STUDENT_ADMIN_KEY_PATH}/student-admin_key* tmp

	# Set proper permissions for directory
	chmod 700 tmp

	# Change into tmp dir
	cd tmp

	# Set proper permissions for key
	chmod 600 student-admin_key*

	# Create a new secure key
	rm -f secure_key*
	ssh-keygen -f secure_key -t ed25519 -N "group14" -C "secure group 14 key"

	# Create new authorized_keys file and insert new key into it
	cat secure_key.pub > authorized_keys
	chmod 600 authorized_keys

	echo "checking to ensure authorized_keys is correct"
	ls -l authorized_keys
	cat authorized_keys

	# Copy new key to .ssh dir
	cp secure_key* $HOME/.ssh/

	# Copy new authorized_keys to server
	scp -i student-admin_key -P ${PORT} -o StrictHostKeyChecking=no authorized_keys student-admin@${MACHINE}:~/.ssh/
fi

# Add key to ssh agent
eval "$(ssh-agent -s)"
ssh-add $HOME/.ssh/secure_key

# Check to ensure proper authorized_keys on remote machine
ssh -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE} "cat ~/.ssh/authorized_keys"

# Clone repository
cd $HOME/tmp
rm -rf CS553_CaseStudy1
git clone https://github.com/pkapoor12/CS553_CaseStudy1.git

# Check if source code exists on remote machine
if ! eval "$COMMAND 'test -d ~/CS553_CaseStudy1'"; then
	# Source code doesn't exist, so copy source code to remote machine
	scp -P ${PORT} -o StrictHostKeyChecking=no -r CS553_CaseStudy1 student-admin@${MACHINE}:~/
else
	# Source code exists, so just sync with remote repo
	${COMMAND} "cd CS553_CaseStudy1 && git pull origin main"
fi
