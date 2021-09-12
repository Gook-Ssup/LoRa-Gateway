import sys
import numpy
import matplotlib.pyplot as plt

k1 = numpy.linspace(0.0, 9.0, 10)

description = "/home/yun/Desktop/practice/"

fig = plt.figure()
# add_subplot : #, 1, sequence
ax = fig.add_subplot(2,1,1)
ax.plot(k1,'r-',lw=1)
ax.set_title('H1')
# Draw Tile
# ax.grid(True)

# Y : sin 4 pi x
# ax.set_ylabel(r'$sin(4 \pi x)$')

# plt.axis : [-x,+x,-y,+y] range
# plt.axis([0,1,-1.5,1.5])

ax = fig.add_subplot(2,1,2)
ax.plot(k1,'b-',lw=1)
ax.set_title('H2')
fig.tight_layout()

fig.savefig(description + 'practice1.png')

