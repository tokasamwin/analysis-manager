from base64 import b64encode
from io import BytesIO
from threading import Event, Thread
from pathlib import Path

import pandas as pd

class routineDashboard(object):
	def __init__(self, routine, columns, output='routine_dashboard.html',
				 refreshSeconds=2, maxRows=100):
		self.routine = routine
		self.columns = list(columns)
		self.output = Path(output)
		self.refreshSeconds = refreshSeconds
		self.maxRows = maxRows
		self._stop = Event()
		self._thread = None

	def _plot(self, frame):
		if frame.empty or not self.columns:
			return ''
		from matplotlib import pyplot as plt
		figure, axis = plt.subplots(figsize=(10, 4))
		for column in self.columns:
			axis.plot(frame.index, frame[column], marker='.', label=column)
		axis.set_xlabel('Routine evaluation')
		axis.legend(loc='best')
		axis.grid(alpha=0.25)
		figure.tight_layout()
		image = BytesIO()
		figure.savefig(image, format='png', dpi=120)
		plt.close(figure)
		return b64encode(image.getvalue()).decode('ascii')

	def refresh(self):
		frame = self.routine.df.copy(deep=True)
		missing = [column for column in self.columns if column not in frame]
		if missing:
			raise KeyError('Dashboard columns are missing from routine.df: {}'.format(
				', '.join(missing)))
		plotFrame = frame[self.columns].apply(pd.to_numeric, errors='coerce').dropna(how='all')
		table = frame.tail(self.maxRows).to_html(index=False, classes='results')
		plot = self._plot(plotFrame)
		name = self.routine.name or self.routine.__class__.__name__
		html = '''<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh}">
<title>{name} dashboard</title>
<style>
body {{ font: 15px sans-serif; margin: 2rem; color: #24303a; }}
table {{ border-collapse: collapse; width: 100%; overflow: auto; }}
th, td {{ border: 1px solid #d5dce0; padding: .4rem .6rem; text-align: right; }}
th {{ background: #edf2f4; position: sticky; top: 0; }}
img {{ max-width: 100%; }}
.table {{ max-height: 28rem; overflow: auto; }}
</style></head><body>
<h1>{name}</h1><p>{rows} rows; updated automatically</p>
{image}<div class="table">{table}</div>
</body></html>'''.format(
			refresh=self.refreshSeconds,
			name=name,
			rows=len(frame),
			image='<img src="data:image/png;base64,{}">'.format(plot) if plot else '',
			table=table)
		self.output.parent.mkdir(parents=True, exist_ok=True)
		self.output.write_text(html, encoding='utf-8')
		return self.output

	def _watch(self):
		while not self._stop.is_set():
			self.refresh()
			self._stop.wait(self.refreshSeconds)

	def start(self):
		self.refresh()
		self._stop.clear()
		self._thread = Thread(target=self._watch, daemon=True)
		self._thread.start()
		return self.output

	def stop(self):
		self._stop.set()
		if self._thread:
			self._thread.join()

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, exceptionType, exception, traceback):
		self.stop()