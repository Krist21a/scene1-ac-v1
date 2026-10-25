"""Publication-independent scientific figure of retained RGB-D geometry."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
root=Path(__file__).resolve().parent
x=json.loads((root/'release-component-diagnostic.json').read_text());c=x['candidate']
points=np.array(c['debug_points'])[:,:2];colors=np.array(c['debug_rgb'])/255.
new=np.array(c['debug_oriented_center']);old=np.array(c['position_base_m'])[:2]
major=np.array(c['major_axis_xy']);basis=np.column_stack((major,[-major[1],major[0]]))
half=np.array(c['oriented_extent_m'])/2
corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
ax.scatter(*(1000*(points-new)).T,c=colors,s=18,edgecolors='#666666',linewidths=.25,label='Same 221 observed depth points')
for center,color,label in [(old,'#ba3045','Previous mixed-frame center'),(new,'#15704f','Center in fitted PCA frame')]:
 ax.add_patch(Polygon(1000*(corners*half@basis.T+center-new),fill=False,edgecolor=color,lw=2,label=label))
 ax.add_patch(Polygon(1000*(corners*(half+.003)@basis.T+center-new),fill=False,edgecolor=color,lw=1,ls='--'))
 ax.plot(*(1000*(center-new)),'x',color=color,ms=8,mew=2)
ax.annotate('',xy=1000*(old-new),xytext=(0,0),arrowprops=dict(arrowstyle='<->',color='#444444'))
ax.text(.98,.98,'Center shift: 13.20 mm\nDimensions unchanged: 109.03 x 45.81 mm\nDashed lines: unchanged 3 mm margin',transform=ax.transAxes,ha='right',va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='#dddddd'))
ax.set(xlabel='Base-frame X relative to fitted center (mm)',ylabel='Base-frame Y relative to fitted center (mm)',title='Release failure: center and dimensions used different frames')
ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=9)
fig.savefig(root/'obb-center-evidence.png',dpi=170);fig.savefig(root/'obb-center-evidence.pdf')
print('obb-center-evidence.png and .pdf written')
