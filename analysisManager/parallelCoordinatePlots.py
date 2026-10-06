import matplotlib.pyplot as plt
import numpy as np

def plotter(df, cols, colourData = None,figsize = (12,6),textIncline=10):
	"""
	Plots parallel coordinates where each axis has its own scale
	without permanent data normalization in the source dataframe.
	"""
	fig, host = plt.subplots(figsize=figsize)
	
	# 1. Setup internal scaling factors
	# We use the first column's range as the 'master' scale for the lines
	ref_col = cols[0]
	ref_min = df[ref_col].min()
	ref_max = df[ref_col].max()
	ref_range = ref_max - ref_min

	# 2. Data checking the colourData:
	if not(colourData):
		colourData = [None] * len(df)

	# 3. Plotting the Lines
	# We map every subsequent column's value into the first column's range
	x = np.arange(len(cols))
	for i, row in df.iterrows():
		line_y = []
		for col in cols:
			# Map value v from [min, max] to [ref_min, ref_max]
			v = row[col]
			c_min, c_max = df[col].min(), df[col].max()
			# Handle zero-division for constant columns
			c_range = (c_max - c_min) if c_max != c_min else 1
			scaled_v = ref_min + (v - c_min) / c_range * ref_range
			line_y.append(scaled_v)
			
		host.plot(x, line_y, color=colourData[i], alpha=0.4)

	# 4. Create and Position Independent Axes
	axes = [host] + [host.twinx() for _ in range(len(cols) - 1)]
	
	for i, ax in enumerate(axes):
		col_name = cols[i]
		ax.set_ylim(df[col_name].min(), df[col_name].max())
		
		# Position the spine horizontally
		ax.spines['left'].set_position(('axes', i / (len(cols) - 1)))
		ax.spines['left'].set_visible(True)
		
		# Strip away unwanted chart junk
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		
		# Labels and Ticks
		ax.set_ylabel(col_name, rotation=textIncline, ha='center', va='center')
		ax.yaxis.set_label_coords(i / (len(cols) - 1), 1.05) # Title above the axis
		
		if i > 0:
			ax.yaxis.set_ticks_position('left')
			ax.toggle_label = True
		else:
			ax.tick_params(axis='y', labelsize=10)

	# Clean up horizontal axis
	host.set_xlim(0, len(cols) - 1)
	host.set_xticks(x)
	host.set_xticklabels([]) # Hide default bottom labels since we put them on top
	host.get_yaxis().set_visible(True)

	fig.tight_layout()
	return fig