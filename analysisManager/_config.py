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

def _defaultDataDir() -> Path:
	return Path(appdirs.user_data_dir(appname='analysisManager'))

def _writeConfig(dataDir: Path) -> None:
	if not(hostdir.exists()): hostdir.mkdir(parents=True, exist_ok=True)
	localConfig = configparser.ConfigParser()
	localConfig['path'] = {
		'dataDir': str(dataDir),
		'configDir': str(hostdir),
		}
	with open(settingsFile, 'w') as f:
		localConfig.write(f)

envDataDir = os.environ.get('ANALYSISMANAGER_DATADIR')

if envDataDir:
	# Non-interactive use (web server, Docker container): the environment
	# variable takes precedence, is used in-memory only, and no prompt is
	# ever shown. The on-disk settings file is left untouched.
	config = configparser.ConfigParser()
	dataDir = Path(envDataDir)
	if not(dataDir.is_dir()): dataDir.mkdir(parents=True, exist_ok=True)
	config['path'] = {
		'dataDir': dataDir,
		'configDir': hostdir,
		}
elif settingsFile.exists():
	config = configparser.ConfigParser()
	config.read(settingsFile)
else:
	config = configparser.ConfigParser()
	dataDir = _defaultDataDir()
	suggestion: Path
	if sys.stdin.isatty():
		print('Settings not detected')
		print('Using default data directory: {}'.format(dataDir))
		interactive = None
		while not(interactive):
			candidate = input('Press enter to accept, or enter a different directory path: ')
			candidatePath = Path(candidate)
			if not len(candidate):
				interactive = dataDir
			elif candidatePath.is_dir():
				interactive = candidatePath
			elif not(candidatePath.exists()) and candidatePath.parent.is_dir():
				candidatePath.mkdir()
				interactive = candidatePath
		suggestion = interactive
	else:
		# Non-interactive use without an environment variable: fall back to
		# the default directory silently so servers and containers do not hang.
		suggestion = dataDir
	config['path'] = {
		'dataDir':Path(suggestion),
		'configDir':hostdir,
		}
	_writeConfig(Path(suggestion))
