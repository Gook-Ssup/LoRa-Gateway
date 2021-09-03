import sys

sys.stdout = open('/home/yun/Desktop/output_signal_in0.txt', 'a')

print(123, end='\t')

sys.stdout.close()