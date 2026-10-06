from shutil import rmtree
import sys, os, json
from pathlib import Path
from string import ascii_uppercase, digits
from random import choice
def randomString(length):
	return ''.join(choice(ascii_uppercase + digits) for _ in range(length))

import appdirs, configparser
from pathlib import Path
hostdir = Path(appdirs.user_config_dir(appname='analysisManager'))
if not(hostdir.parent.exists()): hostdir.parent.mkdir()
settingsFile = hostdir / 'settings.ini'
if settingsFile.exists():
	config = configparser.ConfigParser()
	config.read(settingsFile)
else:
	config = configparser.ConfigParser()
	dataDir = Path(appdirs.user_data_dir(appname='analysisManager'))
	print('Settings not detected')
	print('Using default data directory: {}'.format(dataDir))
	suggestion = None
	while not(suggestion):
		candidate = input('Press enter to accept, or enter a different directory path: ')
		candidatePath = Path(candidate)
		if not len(candidate):
			suggestion = dataDir
		elif candidatePath.is_dir():
			suggestion = candidatePath
		elif not(candidatePath.exists()) and candidatePath.parent.is_dir():
			candidatePath.mkdir()
			suggestion = candidatePath
	config['path'] = {
		'dataDir':Path(suggestion),
		'configDir':hostdir,
		}
	if not hostdir.exists(): hostdir.mkdir()
	with open(hostdir / 'settings.ini','w') as f:
		config.write(f)