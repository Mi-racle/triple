import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt


Fs = 256  # 采样频率 要大于信号频率的两倍可恢复信号
t1 = np.arange(start=0, stop=1, step=1/Fs)  # 1s采样Fs个点

F1 = 50  # 信号1的频率
F2 = 75  # 信号2的频率
y = 2 + 3 * np.cos(2 * np.pi * F1 * t1) + 1.5 * np.cos(2 * np.pi * F2 * t1) + np.random.normal(scale=0.15, size=Fs)

N = len(t1)  # 采样点数

freq = np.arange(N) / N * Fs  # [0, Fs)频率点列表
Y1 = np.fft.fft(y)  # 复数
print('Y[0]', Y1[0])
print('Y[128]', abs(Y1)[120:137])
Y = Y1 / (N / 2)  # 换算成实际的振幅
Y[0] = Y[0] / 2

freq_half = freq[range(int(N / 2))]
Y_half = Y[range(int(N / 2))]

fig, ax = plt.subplots(4, 1, figsize=(12, 12))
ax[0].plot(t1, y)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')

ax[1].plot(freq, abs(Y1), 'r', label='no normalization')
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('Amplitude')
ax[1].legend()

ax[2].plot(freq, abs(Y), 'r', label='normalization')
ax[2].set_xlabel('Freq (Hz)')
ax[2].set_ylabel('Amplitude')
ax[2].set_yticks(np.arange(0, 3))
ax[2].legend()

ax[3].plot(freq_half, abs(Y_half), 'b', label='normalization')
ax[3].set_xlabel('Freq (Hz)')
ax[3].set_ylabel('Amplitude')
ax[3].set_yticks(np.arange(0, 3))
ax[3].legend()

# plt.show()
plt.savefig('a.png')
plt.close()
