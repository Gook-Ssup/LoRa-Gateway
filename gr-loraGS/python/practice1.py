import numpy
import matplotlib.pyplot as plt

k = numpy.linspace(0.0, 1024*8 -1, 1024*8)
arr1 = numpy.zeros(1024*8, dtype=numpy.float) -1
# arr1 = None
arr1[0 : 1024] = k[0:1024]
for i in range(8):
    arr1[i*1024 : (i+1)*1024] = k[i*1024 : (i+1)*1024]
    plt.plot(arr1[i * 1024 : (i+1)*1024])
    # plt.show()
    plt.savefig("/home/yun/test%d.png" %(i + 1))
    plt.clf()
    
arr2 = numpy.zeros(1024, dtype=numpy.float) -1
arr2[0 : 1024] = arr1[1*1024 : 2*1024]
# plt.plot(arr2)
# plt.show()