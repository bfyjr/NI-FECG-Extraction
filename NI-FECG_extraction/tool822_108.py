import numpy as np
from numpy import linalg as la
import math
from scipy import signal as signalTool
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft, istft
from sklearn.decomposition import NMF, PCA
from scipy.fftpack import hilbert
from scipy import interpolate


def get_var(index):  # 获取序列的方差
    temp = []
    for i in range(1, len(index)):
        temp.append(index[i] - index[i - 1])
    return np.var(temp)


def remove_nan(arr):  # 去除信号中的无效值
    for _ in range(3):
        nan_flag = np.isnan(arr)
        for i in range(len(nan_flag)):
            if nan_flag[i] == True and i <= (len(arr) / 2):
                for j in range(1, 20):
                    if nan_flag[i + j] == False:
                        arr[i] = arr[i + j]
                        break
            if nan_flag[i] == True and i > (len(arr) / 2):
                for j in range(1, 20):
                    if nan_flag[i - j] == False:
                        arr[i] = arr[i - j]
                        break
    return arr


def switchICA_envelope(sig, index, M=False, fs=1000, print_coff=False):
    baoluo = getspline(sig)
    # fig, ax = plt.subplots()
    # fig.patch.set_alpha(0.)
    # baoluo2=abs(baoluo[6000:7300])
    # plt.plot(sig[6000:7300],color='black')
    # plt.plot(abs(baoluo2),'r')
    # x = np.linspace(655-250, 655-25, (250-25))
    # x=np.array(x,dtype=int)
    # y = baoluo2[x]
    # plt.plot(x, y,'red')
    # plt.fill_between(x, y, interpolate=True, color='green', alpha=0.5)
    #
    # x = np.linspace(655 + 25, 655+250, (250 - 25))
    # x = np.array(x, dtype=int)
    # y = baoluo2[x]
    # plt.plot(x, y,'red')
    # plt.fill_between(x, y, interpolate=True, color='green', alpha=0.5)
    # plt.axis('off')
    # plt.show()

    if M == True:
        if len(index) < (len(sig) * 0.7 / ((int)(1000 * fs / 1000))):
            return [10000, 0,baoluo]
    else:
        if len(index) < (len(sig) * 1.65 / ((int)(1000 * fs / 1000))) or len(index) >= (
                len(sig) * 3.3 / ((int)(1000 * fs / 1000))):
            return 0
    pos_num = 0  # 使R峰值朝上
    neg_num = 0
    for i in range(len(index)):
        if sig[index[i]] > 0:
            pos_num += 1
        else:
            neg_num += 1
    if pos_num < neg_num:
        sig = sig * -1


    # plt.plot(sig)
    # plt.plot(baoluo,'r')
    # plt.show()

    mean_peak = 0  # 平均峰值强度
    for i in range(len(index)):
        mean_peak += abs(baoluo[index[i]])
    mean_peak /= len(index)
    baoluo = baoluo * 1.0 / mean_peak


    if print_coff:
        plot_sig_annotation(sig, index)
        plt.plot(sig, 'b')
        plt.plot(baoluo * mean_peak, 'y')
        plt.show()
    s = 0.0  # 用每个峰值与周围均值的比值之和来确定信号信噪比高低
    for i in range(len(index)):
        peak = abs(baoluo[index[i]])
        if index[i] - (int(250 * fs / 1000)) < 0 or index[i] + ((int)(250 * fs / 1000)) > len(sig):
            continue
        # 1000hz采样率下，取25-180范围内值
        left = np.mean(abs(baoluo[index[i] - ((int)(250 * fs / 1000)):index[i] - ((int)(25 * fs / 1000))]))
        right = np.mean(abs(baoluo[index[i] + ((int)(25 * fs / 1000)):index[i] + ((int)(250 * fs / 1000))]))
        if left + right > 0:
            s += math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right)))  # 计算峰值强度
            # print(math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right))))
        else:
            s += math.fabs(peak * 1.0 / (mean_peak / 25))  # 计算峰值强度
    # coff = math.fabs(s / (len(sig)))
    coff = s
    coff /= len(index)

    all_num = 0  # 选择峰值周围干扰值较少的
    if M == True:
        for i in range(len(index) - 1):
            ref = (abs(sig[index[i]]) + abs(sig[index[i + 1]])) / 2  # 两个qrs点的值的均值
            for j in sig[index[i] + (int)(45 * fs / 1000):index[i + 1] - (int)(45 * fs / 1000)]:
                if abs(j) > ref * 0.5:
                    all_num += 1
        if len(index) > len(sig) / (int)(180 * fs / 1000):
            all_num = 10000
        return [all_num, coff / var_norm(index), baoluo]
    else:
        temp = []
        for i in range(1, len(index)):
            temp.append(index[i] - index[i - 1])
        for i in range(len(temp)):
            if temp[i] < 0:
                temp[i] = -10000
        temp = np.setdiff1d(temp, np.array([-10000]))
        if print_coff:
            print('coff:', coff, 'var:', np.var(np.diff(index)))
        return coff / np.var(temp)
        # return coff

def get_signal_from_to(aa, bb, cc, dd, aECG_flag):
    sig_from = 0
    sig_to = sig_from + 10000
    delta = 100
    sig_to_list = []
    while sig_to < len(aa):  # 一共6个十秒，每次分析十秒左右的信号
        for i in range(len(aECG_flag)):
            if i == 1:
                temp_sig = [aa, bb, cc, dd][i]
                temp_sig = np.abs(temp_sig)
                max_ = max(temp_sig[sig_to - delta:sig_to + delta])
                while max_ > 0.3:
                    sig_to += delta
                    try:
                        max_ = max(temp_sig[sig_to - delta:sig_to + delta])
                    except:
                        pass
                sig_to_list.append(sig_to)
        # print(sig_from,sig_to)
        sig_from = sig_to
        sig_to = sig_from + 10000
    return sig_to_list


def wavelet_filt(sig):
    wavename = 'bior6.8'
    x = range(0, len(sig))
    w = pywt.Wavelet(wavename)  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(sig), w.dec_len)
    # print('sig_len:',len(sig),' wave_len:',w.dec_len)
    # print("maximum level is " + str(maxlev))
    threshold = 0.34  # Threshold for filtering
    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(sig, wavename, level=maxlev)  # 将信号进行小波分解
    for i in range(1, len(coeffs)):
        threshold = 0.5 - i * 0.05
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波
    datarec = pywt.waverec(coeffs, wavename)  # 重建去噪后的信号
    return datarec


x = np.random.random(size=(6000,))
wavelet_filt(x)


def cancel_MECG_LMS(sig, qrs_index, fs=1000):
    origin_sig = sig.copy()
    mean_dis = np.mean(np.diff(qrs_index))
    template_array = [None] * len(sig)
    if mean_dis < int(500 * fs / 1000):
        left = int(150 * fs / 1000)
        right = int(350 * fs / 1000)
    else:
        left = int(250 * fs / 1000)
        right = int(450 * fs / 1000)
    template = np.zeros((left + right,))
    template_num = 0
    for i in range(len(qrs_index)):  # 两头的不用，用中间的构造模板
        if qrs_index[i] >= left and qrs_index[i] <= len(sig) - right - 1:
            template += sig[qrs_index[i] - left:qrs_index[i] + right]
            template_num += 1
    template = template * 1.0 / template_num
    win = np.ones(len(template))  # 梯形窗
    for i in range(20):
        win[i] = 0.05 * i
        win[len(template) - 1 - i] = 0.05 * i
    template = np.multiply(template, win)
    for j in range(len(qrs_index)):
        index = qrs_index[j]
        if index >= left and index <= len(sig) - right - 1:
            # 参考论文：A robust fetal ECG detection method for abdominal recordings
            M = np.zeros((left + right, 3))
            for i in range(int(200 * fs / 1000)):
                M[:, 0][i] = template[i]
            for i in range(int(200 * fs / 1000), int(300 * fs / 1000)):
                M[:, 1][i] = template[i]
            for i in range(int(300 * fs / 1000), int(700 * fs / 1000)):
                M[:, 2][i] = template[i]
            sig[index - left:index + right] -= template
            template_array[index - left:index + right] = template
            m = sig[index - left:index + right].copy()
            a = np.dot(np.dot(la.inv(np.dot(M.T, M)), M.T), m)

            plt.subplot(211)
            rec = np.dot(M, a)
            plt.plot(rec, 'g', linewidth=0.5)
            plt.plot(sig[index - left:index + right], 'r', linewidth=0.5)
            plt.subplot(212)
            plt.plot(sig[index - left:index + right] - rec, 'g', linewidth=0.5)
            plt.show()

            sig[index - left:index + right] -= (np.dot(M, a))


        else:
            if index < left:
                template_array[0:index + right] = template[left - index:]
                sig[0:index + right] -= template[left - index:]
            if index > len(sig) - right - 1:
                template_array[index - left:] = template[0:len(sig[index - left:])]
                sig[index - left:] -= template[0:len(sig[index - left:])]
    # plt.subplot(211)
    # plt.plot(origin_sig, 'g',linewidth=0.6)
    # plt.plot(template_array, 'r',linewidth=0.6)
    # plt.subplot(212)
    # plt.plot(sig,linewidth=0.4)
    # plt.show()
    return sig


def find_min_index(arr_list, n):
    arr = arr_list.copy()
    min_n_list = []
    for _ in range(n):
        min = 100000000
        pos = 0
        for i in range(len(arr)):
            if arr[i] < min:
                min = arr[i]
                pos = i
        arr[pos] = 1000000
        min_n_list.append(pos)
    return min_n_list


from scipy.fftpack import hilbert


def cancle_MECG_HF(sig, qrs_index, fs=1000):
    mean_dis = np.mean(np.diff(qrs_index))
    if mean_dis < int(500 * fs / 1000):
        left = int(150 * fs / 1000)
        right = int(350 * fs / 1000)
    else:
        left = int(250 * fs / 1000)
        right = int(450 * fs / 1000)
    template = np.zeros((left + right,))
    template_num = 0
    for i in range(len(qrs_index)):  # 两头的不用，用中间的构造模板
        if qrs_index[i] >= left and qrs_index[i] <= len(sig) - right - 1:
            template += sig[qrs_index[i] - left:qrs_index[i] + right]
            template_num += 1
    template = template * 1.0 / template_num
    # # 乘汉宁窗（高斯窗）
    # N = 1000
    # window = list(max(template) * -1 * np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]))
    # template=np.multiply(template,window[int(N/2)-left:int(N/2)+right])
    template_hf = hilbert(template)

    # plt.subplot(311).plot(template)
    # plt.subplot(312).plot(template_hf)
    # plt.subplot(313).plot(hilbert(template_hf))
    # plt.show()

    # print(np.dot(template,template_hf.T))
    for j in range(len(qrs_index)):
        index = qrs_index[j]
        if left <= index <= len(sig) - right - 1:
            fenzi_1 = np.sum(np.multiply(sig[index - left:index + right], template))
            fenmu_1 = np.sum(np.power(template, 2))

            fenzi_2 = np.sum(np.multiply(sig[index - left:index + right], template_hf))
            fenmu_2 = np.sum(np.power(template_hf, 2))

            coeff1 = fenzi_1 / fenmu_1
            coeff2 = fenzi_2 / fenmu_2
            rec = coeff1 * template + coeff2 * template_hf

            # plt.subplot(311)
            # plt.plot(template, 'g',linewidth=0.5)
            # plt.plot(template_hf, 'r',linewidth=0.5)
            # plt.title('从上到下：模板(绿)及其希尔伯特变换(红)；  原信号(绿)及重建母体信号(红)；  去除母体信号剩余胎儿信号')
            # plt.subplot(312)
            # plt.plot(sig[index - left:index + right], 'g',linewidth=0.5)
            # plt.plot(rec, 'r',linewidth=0.5)
            # # plt.title('重建母体信号及原信号')
            # plt.subplot(313)
            # # plt.title('去除母体信号剩余胎儿信号')
            # plt.plot(sig[index - left:index + right]-rec,linewidth=0.5)
            # plt.show()

            sig[index - left:index + right] -= rec
        elif index < left:
            fenzi_1 = np.sum(np.multiply(sig[0:index + right], template[left - index:]))
            fenmu_1 = np.sum(np.power(template[left - index:], 2))

            fenzi_2 = np.sum(np.multiply(sig[0:index + right], template_hf[left - index:]))
            fenmu_2 = np.sum(np.power(template_hf[left - index:], 2))

            rec = (fenzi_1 / fenmu_1) * template[left - index:] + (fenzi_2 / fenmu_2) * template_hf[left - index:]
            sig[0:index + right] -= rec
        else:
            fenzi_1 = np.sum(np.multiply(sig[index - left:len(sig)], template[0:left + len(sig) - index]))
            fenmu_1 = np.sum(np.power(template[0:left + len(sig) - index], 2))

            fenzi_2 = np.sum(np.multiply(sig[index - left:len(sig)], template_hf[0:left + len(sig) - index]))
            fenmu_2 = np.sum(np.power(template_hf[0:left + len(sig) - index], 2))

            rec = (fenzi_1 / fenmu_1) * template[0:left + len(sig) - index] + (fenzi_2 / fenmu_2) * template_hf[
                                                                                                    0:left + len(
                                                                                                        sig) - index]
            sig[index - left:len(sig)] -= rec
    return sig


def get_orth(sig):
    pi = 3.14
    f_a, t_a, zxx = stft(sig, fs=1000, nperseg=10, return_onesided=True, noverlap=9)
    zxx *= np.exp(1j * (pi / 2))
    t, rec = istft(zxx, fs=1000, nperseg=10, noverlap=9)
    return rec


def cancle_MECG_HF_2(sig, qrs_index, fs=1000):
    mean_dis = np.mean(np.diff(qrs_index))
    if mean_dis < int(500 * fs / 1000):
        left = int(150 * fs / 1000)
        right = int(350 * fs / 1000)
    else:
        left = int(250 * fs / 1000)
        right = int(450 * fs / 1000)
    template = np.zeros((left + right,))
    template_num = 0
    for i in range(len(qrs_index)):  # 两头的不用，用中间的构造模板
        if qrs_index[i] >= left and qrs_index[i] <= len(sig) - right - 1:
            template += sig[qrs_index[i] - left:qrs_index[i] + right]
            template_num += 1
    template = template * 1.0 / template_num
    # # 乘汉宁窗（高斯窗）
    # N = 1000
    # window = list(max(template) * -1 * np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]))
    # template=np.multiply(template,window[int(N/2)-left:int(N/2)+right])
    template_hf = get_orth(template)

    # plt.subplot(311).plot(template)
    # plt.subplot(312).plot(template_hf)
    # plt.subplot(313).plot(hilbert(template_hf))
    # plt.show()

    # print(np.dot(template,template_hf.T))
    for j in range(len(qrs_index)):
        index = qrs_index[j]
        if left <= index <= len(sig) - right - 1:
            fenzi_1 = np.sum(np.multiply(sig[index - left:index + right], template))
            fenmu_1 = np.sum(np.power(template, 2))

            fenzi_2 = np.sum(np.multiply(sig[index - left:index + right], template_hf))
            fenmu_2 = np.sum(np.power(template_hf, 2))

            coeff1 = fenzi_1 / fenmu_1
            coeff2 = fenzi_2 / fenmu_2
            rec = coeff1 * template + coeff2 * template_hf

            # plt.subplot(311)
            # plt.plot(template, 'g',linewidth=0.5)
            # plt.plot(template_hf, 'r',linewidth=0.5)
            # plt.title('从上到下：模板(绿)及其希尔伯特变换(红)；  原信号(绿)及重建母体信号(红)；  去除母体信号剩余胎儿信号')
            # plt.subplot(312)
            # plt.plot(sig[index - left:index + right], 'g',linewidth=0.5)
            # plt.plot(rec, 'r',linewidth=0.5)
            # # plt.title('重建母体信号及原信号')
            # plt.subplot(313)
            # # plt.title('去除母体信号剩余胎儿信号')
            # plt.plot(sig[index - left:index + right]-rec,linewidth=0.5)
            # plt.show()

            sig[index - left:index + right] -= rec
        elif index < left:
            fenzi_1 = np.sum(np.multiply(sig[0:index + right], template[left - index:]))
            fenmu_1 = np.sum(np.power(template[left - index:], 2))

            fenzi_2 = np.sum(np.multiply(sig[0:index + right], template_hf[left - index:]))
            fenmu_2 = np.sum(np.power(template_hf[left - index:], 2))

            rec = (fenzi_1 / fenmu_1) * template[left - index:] + (fenzi_2 / fenmu_2) * template_hf[left - index:]
            sig[0:index + right] -= rec
        else:
            fenzi_1 = np.sum(np.multiply(sig[index - left:len(sig)], template[0:left + len(sig) - index]))
            fenmu_1 = np.sum(np.power(template[0:left + len(sig) - index], 2))

            fenzi_2 = np.sum(np.multiply(sig[index - left:len(sig)], template_hf[0:left + len(sig) - index]))
            fenmu_2 = np.sum(np.power(template_hf[0:left + len(sig) - index], 2))

            rec = (fenzi_1 / fenmu_1) * template[0:left + len(sig) - index] + (fenzi_2 / fenmu_2) * template_hf[
                                                                                                    0:left + len(
                                                                                                        sig) - index]
            sig[index - left:len(sig)] -= rec
    return sig


def filt(a, low_freq=None, high_freq=None):
    if low_freq == None:
        low_freq = 0.015
    if high_freq == None:
        high_freq = 0.15
    w1, w2 = signalTool.butter(4, [low_freq, high_freq], 'bandpass')
    w3, w4 = signalTool.iirnotch(50, 50, fs=1000)  # 50hz陷波滤波器
    w5, w6 = signalTool.iirnotch(60, 50, fs=1000)  # 60hz陷波滤波器
    a = signalTool.filtfilt(w1, w2, a.T)
    a = signalTool.filtfilt(w3, w4, a.T)
    a = signalTool.filtfilt(w5, w6, a.T)
    return a


def plot_sig_annotation(sig, annList, title=None, color='blue', qrsmark='r+',axis_off=False):
    annArr = np.array([None] * len(sig)).T
    for i in range(len(annList)):
        annArr[annList[i]] = sig[annList[i]]
    plt.figure()
    plt.plot(sig, color, linewidth=1)
    plt.plot(annArr, qrsmark)
    if axis_off:
        plt.axis('off')
    if title != None:
        plt.title(title)
    plt.show()


def switchICA(sig, index, M=False, fs=1000, print_coff=False):
    if M == True:
        if len(index) < (len(sig) * 0.7 / ((int)(1000 * fs / 1000))):
            return [10000, 0]
    else:
        # print('switchICA:',len(index))
        # if len(index)<(len(sig)*1.65/((int)(1000*fs/1000))) or len(index)>=(len(sig)*3/((int)(1000*fs/1000))): # 8.2更改
        if len(index) < (len(sig) * 1.65 / ((int)(1000 * fs / 1000))) or len(index) >= (
                len(sig) * 3.3 / ((int)(1000 * fs / 1000))):
            return 0
    # sig=sig*1.0/max(sig)
    mean_peak = 0  # 平均峰值强度
    for i in range(len(index)):
        mean_peak += (abs(sig[index[i]]))
    mean_peak /= len(index)
    sig = sig * 1.0 / mean_peak
    s = 0.0  # 用每个峰值与周围均值的比值之和来确定信号信噪比高低
    for i in range(len(index)):
        peak = math.fabs(sig[index[i]])
        if index[i] - (int(250 * fs / 1000)) < 0 or index[i] + ((int)(250 * fs / 1000)) > len(sig):
            continue
        # 1000hz采样率下，取25-180范围内值
        left = np.mean(np.abs(sig[index[i] - ((int)(250 * fs / 1000)):index[i] - ((int)(25 * fs / 1000))]))
        right = np.mean(np.abs(sig[index[i] + ((int)(25 * fs / 1000)):index[i] + ((int)(250 * fs / 1000))]))
        if left + right > 0:
            s += math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right)))  # 计算峰值强度
            # print(math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right))))
        else:
            s += math.fabs(peak * 1.0 / (mean_peak / 25))  # 计算峰值强度
    coff = math.fabs(s / (len(sig)))

    all_num = 0  # 选择峰值周围干扰值较少的
    if M == True:
        for i in range(len(index) - 1):
            ref = (abs(sig[index[i]]) + abs(sig[index[i + 1]])) / 2  # 两个qrs点的值的均值
            for j in sig[index[i] + (int)(35 * fs / 1000):index[i + 1] - (int)(35 * fs / 1000)]:
                if abs(j) > ref * 0.5:
                    all_num += 1
        if len(index) > len(sig) / (int)(180 * fs / 1000):
            all_num = 10000
        return [all_num, coff]
    else:
        temp = []
        for i in range(1, len(index)):
            temp.append(index[i] - index[i - 1])
        if print_coff:
            print('coff:', coff, 'var:', np.var(np.diff(index)))
        return coff / np.var(temp)
        # return coff


def findpeaks(x,low):
    if low:return signalTool.argrelextrema(x, np.less_equal)[0]
    return signalTool.argrelextrema(x, np.greater_equal)[0]


# 判断当前的序列是否为 IMF 序列
def isImf(x):
    N = np.size(x)
    pass_zero = np.sum(x[0:N - 2] * x[1:N - 1] < 0)  # 过零点的个数
    peaks_num = np.size(findpeaks(x)) + np.size(findpeaks(-x))  # 极值点的个数
    if abs(pass_zero - peaks_num) > 1:
        return False
    else:
        return True


# 获取当前样条曲线
def getspline(x,low=False):
    N = np.size(x)
    peaks = findpeaks(x,low)
    if (len(peaks) <= 3):
        if (len(peaks) < 2):
            peaks = np.concatenate(([0], peaks))
            peaks = np.concatenate((peaks, [N - 1]))  # 这里是为了防止样条次数不够，无法插值的情况
        t = interpolate.splrep(peaks, y=x[peaks], w=None, xb=None, xe=None, k=len(peaks) - 1)
        return interpolate.splev(np.arange(N), t)
    t = interpolate.splrep(peaks, y=x[peaks])
    return interpolate.splev(np.arange(N), t)





def var_norm(index1):
    diff = np.diff(index1)
    mean = np.mean(diff)
    for i in range(len(diff)):
        diff[i] = diff[i] / mean
    return np.var(diff)


def switchICA_envelope_var_norm(sig, index, M=False, fs=1000, print_coff=False):
    if M == True:
        if len(index) < (len(sig) * 0.7 / ((int)(1000 * fs / 1000))):
            return [10000, 0]
    else:
        # print('switchICA:',len(index))
        # if len(index)<(len(sig)*1.65/((int)(1000*fs/1000))) or len(index)>=(len(sig)*3/((int)(1000*fs/1000))): # 8.2更改
        if len(index) < (len(sig) * 1.65 / ((int)(1000 * fs / 1000))) or len(index) >= (
                len(sig) * 3.3 / ((int)(1000 * fs / 1000))):
            return 0
    # sig=sig*1.0/max(sig)

    pos_num = 0  # 使R峰值朝上
    neg_num = 0
    for i in range(len(index)):
        if sig[index[i]] > 0:
            pos_num += 1
        else:
            neg_num += 1
    if pos_num < neg_num:
        sig = sig * -1

    baoluo = getspline(sig)

    mean_peak = 0  # 平均峰值强度
    for i in range(len(index)):
        mean_peak += abs(baoluo[index[i]])
    mean_peak /= len(index)
    baoluo = baoluo * 1.0 / mean_peak
    if print_coff:
        plot_sig_annotation(sig, index)
        plt.plot(sig, 'b')
        plt.plot(baoluo * mean_peak, 'y')
        plt.show()
    s = 0.0  # 用每个峰值与周围均值的比值之和来确定信号信噪比高低
    for i in range(len(index)):
        peak = abs(baoluo[index[i]])
        if index[i] - (int(250 * fs / 1000)) < 0 or index[i] + ((int)(250 * fs / 1000)) > len(sig):
            continue
        # 1000hz采样率下，取25-180范围内值
        left = np.mean(abs(baoluo[index[i] - ((int)(250 * fs / 1000)):index[i] - ((int)(25 * fs / 1000))]))
        right = np.mean(abs(baoluo[index[i] + ((int)(25 * fs / 1000)):index[i] + ((int)(250 * fs / 1000))]))
        if left + right > 0:
            s += math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right)))  # 计算峰值强度
            # print(math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right))))
        else:
            s += math.fabs(peak * 1.0 / (mean_peak / 25))  # 计算峰值强度
    coff = math.fabs(s / (len(sig)))
    coff = s

    all_num = 0  # 选择峰值周围干扰值较少的
    if M == True:
        for i in range(len(index) - 1):
            ref = (abs(sig[index[i]]) + abs(sig[index[i + 1]])) / 2  # 两个qrs点的值的均值
            for j in sig[index[i] + (int)(35 * fs / 1000):index[i + 1] - (int)(35 * fs / 1000)]:
                if abs(j) > ref * 0.5:
                    all_num += 1
        if len(index) > len(sig) / (int)(180 * fs / 1000):
            all_num = 10000
        return [all_num, coff]
    else:
        if print_coff:
            print('coff:', coff, 'var:', np.var(np.diff(index)))
        return coff / var_norm(index)
        # return coff


def switchICA_envelope_change_dist(sig, index, M=False, fs=1000, print_coff=False):
    if M == True:
        if len(index) < (len(sig) * 0.7 / ((int)(1000 * fs / 1000))):
            return [10000, 0]
    else:
        # print('switchICA:',len(index))
        # if len(index)<(len(sig)*1.65/((int)(1000*fs/1000))) or len(index)>=(len(sig)*3/((int)(1000*fs/1000))): # 8.2更改
        if len(index) < (len(sig) * 1.65 / ((int)(1000 * fs / 1000))) or len(index) >= (
                len(sig) * 3.3 / ((int)(1000 * fs / 1000))):
            return 0
    # sig=sig*1.0/max(sig)

    pos_num = 0  # 使R峰值朝上
    neg_num = 0
    for i in range(len(index)):
        if sig[index[i]] > 0:
            pos_num += 1
        else:
            neg_num += 1
    if pos_num < neg_num:
        sig = sig * -1

    baoluo = getspline(sig)

    mean_peak = 0  # 平均峰值强度
    for i in range(len(index)):
        mean_peak += abs(baoluo[index[i]])
    mean_peak /= len(index)
    baoluo = baoluo * 1.0 / mean_peak
    if print_coff:
        plot_sig_annotation(sig, index)
        plt.plot(sig, 'b')
        plt.plot(baoluo * mean_peak, 'y')
        plt.show()
    s = 0.0  # 用每个峰值与周围均值的比值之和来确定信号信噪比高低
    for i in range(len(index)):
        peak = abs(baoluo[index[i]])
        if index[i] - (int(290 * fs / 1000)) < 0 or index[i] + ((int)(290 * fs / 1000)) > len(sig):
            continue
        # 1000hz采样率下，取25-180范围内值
        left = np.mean(abs(baoluo[index[i] - ((int)(290 * fs / 1000)):index[i] - ((int)(25 * fs / 1000))]))
        right = np.mean(abs(baoluo[index[i] + ((int)(25 * fs / 1000)):index[i] + ((int)(290 * fs / 1000))]))
        if left + right > 0:
            s += math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right)))  # 计算峰值强度
            # print(math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right))))
        else:
            s += math.fabs(peak * 1.0 / (mean_peak / 25))  # 计算峰值强度
    # coff = math.fabs(s / (len(sig)))
    coff = s

    all_num = 0  # 选择峰值周围干扰值较少的
    if M == True:
        for i in range(len(index) - 1):
            ref = (abs(sig[index[i]]) + abs(sig[index[i + 1]])) / 2  # 两个qrs点的值的均值
            for j in sig[index[i] + (int)(35 * fs / 1000):index[i + 1] - (int)(35 * fs / 1000)]:
                if abs(j) > ref * 0.5:
                    all_num += 1
        if len(index) > len(sig) / (int)(180 * fs / 1000):
            all_num = 10000
        return [all_num, coff]
    else:
        temp = []
        for i in range(1, len(index)):
            temp.append(index[i] - index[i - 1])
        if print_coff:
            print('coff:', coff, 'var:', np.var(np.diff(index)))
        return coff / np.var(temp)
        # return coff


def switchICA_envelope2(sig, index, M=False, fs=1000, print_coff=False):  # 只选取大于零的值
    if M == True:
        if len(index) < (len(sig) * 0.7 / ((int)(1000 * fs / 1000))):
            return [10000, 0]
    else:
        # print('switchICA:',len(index))
        # if len(index)<(len(sig)*1.65/((int)(1000*fs/1000))) or len(index)>=(len(sig)*3/((int)(1000*fs/1000))): # 8.2更改
        if len(index) < (len(sig) * 1.65 / ((int)(1000 * fs / 1000))) or len(index) >= (
                len(sig) * 3.3 / ((int)(1000 * fs / 1000))):
            return 0
    # sig=sig*1.0/max(sig)

    pos_num = 0  # 使R峰值朝上
    neg_num = 0
    for i in range(len(index)):
        if sig[index[i]] > 0:
            pos_num += 1
        else:
            neg_num += 1
    if pos_num < neg_num:
        sig = sig * -1

    baoluo = getspline(sig)

    mean_peak = 0  # 平均峰值强度
    for i in range(len(index)):
        mean_peak += abs(baoluo[index[i]])
    mean_peak /= len(index)
    baoluo = baoluo * 1.0 / mean_peak
    if print_coff:
        plot_sig_annotation(sig, index)
        plt.plot(sig, 'b')
        plt.plot(baoluo * mean_peak, 'y')
        plt.show()
    s = 0.0  # 用每个峰值与周围均值的比值之和来确定信号信噪比高低
    for i in range(len(index)):
        peak = abs(baoluo[index[i]])
        if index[i] - (int(250 * fs / 1000)) < 0 or index[i] + ((int)(250 * fs / 1000)) > len(sig):
            continue
        # 1000hz采样率下，取25-180范围内值

        temp_left = []
        for j in baoluo[index[i] - ((int)(250 * fs / 1000)):index[i] - ((int)(25 * fs / 1000))]:
            if j > 0:
                temp_left.append(j)
        left = np.mean(temp_left)

        temp_right = []
        for j in baoluo[index[i] + ((int)(25 * fs / 1000)):index[i] + ((int)(250 * fs / 1000))]:
            if j > 0:
                temp_right.append(j)
        right = np.mean(temp_right)
        if left + right > 0:
            s += math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right)))  # 计算峰值强度
            # print(math.fabs((peak * 1.0 / (left + right)) * (peak * 1.0 / (left + right))))
        else:
            s += math.fabs(peak * 1.0 / (mean_peak / 25))  # 计算峰值强度
    # coff = math.fabs(s / (len(sig)))
    coff = s

    all_num = 0  # 选择峰值周围干扰值较少的
    if M == True:
        for i in range(len(index) - 1):
            ref = (abs(sig[index[i]]) + abs(sig[index[i + 1]])) / 2  # 两个qrs点的值的均值
            for j in sig[index[i] + (int)(35 * fs / 1000):index[i + 1] - (int)(35 * fs / 1000)]:
                if abs(j) > ref * 0.5:
                    all_num += 1
        if len(index) > len(sig) / (int)(180 * fs / 1000):
            all_num = 10000
        return [all_num, coff]
    else:
        temp = []
        for i in range(1, len(index)):
            temp.append(index[i] - index[i - 1])
        if print_coff:
            print('coff:', coff, 'var:', np.var(np.diff(index)))
        return coff / np.var(temp)
        # return coff


def is_channel_qualified(sig, length):
    peroid_num = int(length / 1000)
    err_peroid = 0
    for i in range(peroid_num):
        err_point = 0
        p = abs(sig[i * 1000:(i + 1) * 1000])
        max_value = max(p)
        for j in p:
            if j > (0.4 * max_value):
                err_point += 1
        if err_point >= 100:
            err_peroid += 1
    if err_peroid >= (int)(peroid_num * 0.7):
        return False
    else:
        return True


def switch(sig, index):
    if len(index) < len(sig) / 1300:
        return 0
    max_value = max(abs(min(sig)), abs(max(sig)))
    sig = sig * 1.0 / max_value
    s = 0.0
    num = 0
    for i in range(1, len(index) - 1):
        peak = math.fabs(sig[index[i]])
        if index[i] - 200 < 0 or index[i] + 200 > len(sig):
            continue
        left = np.mean(np.abs(sig[index[i] - 200:index[i] - 170]))
        right = np.mean(np.abs(sig[index[i] + 170:index[i] + 200]))
        if left + right > 0:
            s += math.fabs(peak * 1.0 / (left + right))  # 计算峰值强度
        else:
            s += math.fabs(peak * 1.0 / (left + right + peak / 15))  # 计算峰值强度
        num += 1
    if num == 0: return 0
    coff = math.fabs(s / (num)) / max_value
    return coff


def switch_119(sig, index):
    if len(index) < len(sig) * 0.95 / 1000 or len(index) > len(sig) * 3.4 / 1000:
        return 0
    max_value = np.max(np.abs(sig))
    coff = 0
    for i in range(len(index)):
        coff += abs(sig[index[i]]) / max_value
        # print(coff)
    return coff


def switch_117(sig, index):
    if len(index) < len(sig) / 1300:
        return 0
    max_value = max(abs(min(sig)), abs(max(sig)))
    sig = sig * 1.0 / max_value
    s = 0.0
    num = 0
    for i in range(1, len(index) - 1):
        peak = math.fabs(sig[index[i]])
        if index[i] - 100 < 0 or index[i] + 100 > len(sig):
            continue
        left = np.mean(np.abs(sig[index[i] - 100:index[i] - 30]))
        right = np.mean(np.abs(sig[index[i] + 30:index[i] + 100]))
        if left + right > 0:
            s += math.fabs(peak * 1.0 / (left + right))  # 计算峰值强度
        else:
            s += math.fabs(peak * 1.0 / (left + right + peak / 15))  # 计算峰值强度
        num += 1
    if num == 0: return 0
    coff = math.fabs(s / num)
    return coff


def check_polarity(sig, fs=1000, F=False):  # 改变信号极性，使母体R峰朝上
    win_size = int(1000 * fs / 1000)
    if F:
        win_size = int(500 * fs / 1000)
    pos_list = []
    negative_list = []
    for i in range(int(len(sig) / win_size)):
        start = i * win_size
        end = (i + 1) * win_size if (i + 1) * win_size < len(sig) else len(sig)
        max_value = np.max(sig[start:end])
        min_value = np.min(sig[start:end])
        pos_list.append(abs(max_value))
        negative_list.append(abs(min_value))
    pos_list = np.sort(pos_list)
    negative_list = np.sort(negative_list)
    discard_num = int(len(pos_list) * 20.0 / 100)  # 丢弃前百分之二的峰值，避免脉冲干扰
    sum_pos = np.sum(pos_list[:-1 * discard_num])
    sum_neg = np.sum(negative_list[:-1 * discard_num])
    if sum_neg > sum_pos:
        sig = sig * -1
    return sig


def correct_qrs_inds(sig, index, is_print=False, pos_flag=True, fs=1000):
    length = len(sig)
    if len(index) == 0:
        # print('qrs定位数目为0，退出')
        raise Exception

    for i in range(2):
        sum = 0.0
        for i in range(len(index)):
            sum += math.fabs(sig[index[i]])
        mean = sum * 1.0 / len(index)
        for i in range(len(index)):
            if math.fabs(sig[index[i]]) < 0.4 * mean:  # 去除幅度不够的定位点#0.65
                index[i] = -1
        a = np.array([-1])
        index = np.setdiff1d(index, a)  # 去除上一步所有错误点
    dis_arr = []
    for i in range(len(index) - 1):
        dis_arr.append((index[i + 1] - index[i]))
    try:
        mean_dis = (int)(np.mean(dis_arr))
    except:
        return index
    for hh in range(5):  # 一个周期内识别出多个QRS点,最多找出5个
        for i in range(1, len(index) - 2):
            if index[i + 1] - index[i] < mean_dis * 0.65:
                var1 = pow(index[i] - index[i - 1], 2) + pow(index[i + 2] - index[i], 2)
                var2 = pow(index[i + 1] - index[i - 1], 2) + pow(index[i + 2] - index[i + 1], 2)
                if var1 < var2:
                    index[i + 1] = -1
                else:
                    index[i] = -1
                break
        a = np.array([-1])
        index = np.setdiff1d(index, a)  # 去除上一步所有错误点
        if len(index) > 2:
            if index[1] - index[0] < (int)(300 * fs / 1000):
                if abs(index[2] - index[1] - mean_dis) < abs(index[2] - index[0] - mean_dis):
                    index[0] = -1
                else:
                    index[1] = -1
            a = np.array([-1])
            index = np.setdiff1d(index, a)  # 去除上一步所有错误点

    dis_arr = []
    for i in range(len(index) - 1):
        dis_arr.append((index[i + 1] - index[i]))
    try:
        mean_dis = (int)(np.mean(dis_arr))
    except:
        return index
    # print(index)
    # 补充第一个可能漏掉的点
    for m in range(2):
        max = 0
        pos = 0
        end = (int)(mean_dis * 0.8)
        if index[0] <= end:  # 300
            pass
        else:
            if pos_flag == True:
                pos = np.argmax(sig[0:end])
            else:
                pos = np.argmin(sig[0:end])
            if math.fabs(sig[pos]) >= 0.2 * mean:
                index = np.insert(index, 0, pos)
            if math.fabs(sig[pos]) >= 0.2 * mean:
                index = np.insert(index, 0, pos)

    dis_arr = []
    for i in range(len(index) - 1):
        dis_arr.append((index[i + 1] - index[i]))
    try:
        mean_dis = (int)(np.mean(dis_arr))
    except:
        return index
    # 间距较大，可能漏掉，插值
    for _ in range(5):
        for i in range(len(index) - 1):
            this = index[i]
            next = index[i + 1]
            if next - this >= mean_dis * 1.5:
                p = this + (int)(60 * fs / 1000)
                q = next - (int)(60 * fs / 1000)
                if p >= q:
                    continue
                if pos_flag == True:
                    p += np.argmax(sig[p:q])
                else:
                    p += np.argmin(sig[p:q])
                index = np.insert(index, i + 1, p)
        # 计算平均间距

    dis_arr = []
    for i in range(len(index) - 1):
        dis_arr.append((index[i + 1] - index[i]))
    try:
        mean_dis = (int)(np.mean(dis_arr))
    except:
        return index
    for i in range(2, len(index) - 1):
        if index[i] - index[i - 1] < mean_dis / 2:  # 一个周期内识别出多个QRS点
            left = index[i - 2]
            right = index[i + 1]
            ms1 = math.pow(math.fabs(index[i - 1] - left) - mean_dis, 2) + math.pow(
                math.fabs(index[i - 1] - right) - mean_dis, 2)
            ms2 = math.pow(math.fabs(index[i] - left) - mean_dis, 2) + math.pow(math.fabs(index[i] - right) - mean_dis,
                                                                                2)
            if ms1 < ms2:
                index[i] = -1
            else:
                index[i - 1] = -1
            continue
    a = np.array([-1])
    index = np.setdiff1d(index, a)  # 去除上一步所有错误点

    # print('index:',index)
    for m in range(3):
        # 补充第一个可能的R点
        sum = 0.0
        for i in range(len(index)):
            sum += math.fabs(sig[index[i]])
        mean = sum * 1.0 / len(index)

        dis12 = (int)(mean_dis * 1.2)
        if index[0] > dis12:
            if pos_flag == True:
                pos = np.argmax(sig[0:mean_dis])
            else:
                pos = np.argmin(sig[0:mean_dis])
            if abs(sig[pos]) > 0.5 * mean:
                index = np.insert(index, 0, pos)
        # 补充最后一个可能的R点
        if length - index[-1] > dis12 * 0.7:
            dis12 = int(dis12 * 0.7)
            if pos_flag == True:
                pos = np.argmax(sig[length - dis12:]) + length - dis12
            else:
                pos = np.argmin(sig[length - dis12:]) + length - dis12
            if math.fabs(sig[pos]) > mean * 0.1:
                index = np.insert(index, len(index), pos)
    index = list(index)

    mean_dis = int(np.mean(np.diff(index)))
    for i in range(len(index) - 1):
        if i >= len(index) - 1:
            break
        if index[i + 1] - index[i] < 0.2 * mean_dis:
            if i < len(index) / 2:
                dist1 = abs(index[i + 2] - index[i] - mean_dis)
                dist2 = abs(index[i + 2] - index[i + 1] - mean_dis)
                if dist1 < dist2:
                    index.pop(i + 1)
                else:
                    index.pop(i)
            else:
                dist1 = abs(index[i] - index[i - 1] - mean_dis)
                dist2 = abs(index[i + 1] - index[i - 1] - mean_dis)
                if dist1 < dist2:
                    index.pop(i + 1)
                else:
                    index.pop(i)

    return index


def correct_qrs_F(sig, index, is_print=False, pos_flag=True, fs=1000):
    length = len(sig)
    if len(index) <= len(sig) * 0.6 / ((int)(1000 * fs / 1000)):
        return []
    if is_print == True:
        print('初始index:', index)

    for i in range(6):
        sum = 0.0
        for i in range(len(index)):
            sum += math.fabs(sig[index[i]])
        mean = sum * 1.0 / len(index)
        for i in range(len(index)):
            if math.fabs(sig[index[i]]) < 0.2 * mean:  # 去除幅度不够的定位点#0.65
                index[i] = -1
        a = np.array([-1])
        index = np.setdiff1d(index, a)  # 去除上一步所有错误点

    if is_print == True:
        print('去除幅度不够后index:', index)
    dis_arr = []
    for i in range(len(index) - 1):
        dis_arr.append((index[i + 1] - index[i]))
    try:
        mean_dis = (int)(np.mean(dis_arr))
    except:
        return index
    for hh in range(5):  # 一个周期内识别出多个QRS点,最多找出5个
        for i in range(1, len(index) - 2):
            if index[i + 1] - index[i] < (int)(300 * fs / 1000):
                var1 = pow(index[i] - index[i - 1], 2) + pow(index[i + 2] - index[i], 2)
                var2 = pow(index[i + 1] - index[i - 1], 2) + pow(index[i + 2] - index[i + 1], 2)
                if var1 < var2:
                    index[i + 1] = -1
                else:
                    index[i] = -1
                break
        a = np.array([-1])
        index = np.setdiff1d(index, a)  # 去除上一步所有错误点
        if len(index) > 2:
            if index[1] - index[0] < (int)(300 * fs / 1000):
                if abs(index[2] - index[1] - mean_dis) < abs(index[2] - index[0] - mean_dis):
                    index[0] = -1
                else:
                    index[1] = -1
            a = np.array([-1])
            index = np.setdiff1d(index, a)  # 去除上一步所有错误点

    if is_print == True:
        print('去除一个周期内多个r点后的dindex:', index)
    dis_arr = []
    for i in range(len(index) - 1):
        dis_arr.append((index[i + 1] - index[i]))
    try:
        mean_dis = (int)(np.mean(dis_arr))
    except:
        return index
    # print(index)
    # 补充第一个可能漏掉的点
    for m in range(4):
        max = 0
        pos = 0
        if index[0] <= mean_dis * 0.8:  # 300
            pass
        else:
            if index[0] <= (int)(280 * fs / 1000):
                continue
            if pos_flag == True:
                pos = np.argmax(sig[0:index[0] - (int)(280 * fs / 1000)])
            else:
                pos = np.argmin(sig[0:index[0] - (int)(280 * fs / 1000)])
            if math.fabs(sig[pos]) >= 0.3 * mean:
                index = np.insert(index, 0, pos)
            # if math.fabs(sig[pos]) >= 0.3 * mean:
            #     index = np.insert(index, 0, pos)

    dis_arr = []
    for i in range(len(index) - 1):
        dis_arr.append((index[i + 1] - index[i]))
    try:
        mean_dis = (int)(np.mean(dis_arr))
    except:
        return index
    # print(mean_dis,index)
    # 间距较大，可能漏掉，插值
    for _ in range(6):
        for i in range(len(index) - 1):
            this = index[i]
            next = index[i + 1]
            if next - this >= mean_dis * 1.2:
                p = this + (int)(40 * fs / 1000)
                q = next - (int)(40 * fs / 1000)
                if p >= q:
                    continue
                if pos_flag == True:
                    p += np.argmax(sig[p:q])
                else:
                    p += np.argmin(sig[p:q])
                index = np.insert(index, i + 1, p)
        # 计算平均间距
    if is_print == True:
        print('间距较大插值后的index:', index)
    dis_arr = []
    for i in range(len(index) - 1):
        dis_arr.append((index[i + 1] - index[i]))
    try:
        mean_dis = (int)(np.mean(dis_arr))
    except:
        return index
    # 在开头结尾和中间去除隔得很近的点
    if index[1] - index[0] < mean_dis / 2:
        if abs(index[2] - index[1] - mean_dis) < abs(index[2] - index[0] - mean_dis):
            index[0] = -1
        else:
            index[1] = -1
    if index[-1] - index[-2] < mean_dis / 2:
        if abs(index[-2] - index[-3] - mean_dis) < abs(index[-1] - index[-3] - mean_dis):
            index[-1] = -1
        else:
            index[-2] = -1
    a = np.array([-1])
    index = np.setdiff1d(index, a)  # 去除上一步所有错误点
    if is_print == True:
        print('开头结尾和中间去除隔得很近的点后index:', index)
    for _ in range(4):
        for i in range(2, len(index) - 1):
            if index[i] - index[i - 1] < mean_dis / 2:  # 一个周期内识别出多个QRS点
                left = index[i - 2]
                right = index[i + 1]
                ms1 = math.pow(math.fabs(index[i - 1] - left) - mean_dis, 2) + math.pow(
                    math.fabs(index[i - 1] - right) - mean_dis, 2)
                ms2 = math.pow(math.fabs(index[i] - left) - mean_dis, 2) + math.pow(
                    math.fabs(index[i] - right) - mean_dis,
                    2)
                if ms1 < ms2:
                    index[i] = -1
                else:
                    index[i - 1] = -1
                continue
        a = np.array([-1])
        index = np.setdiff1d(index, a)  # 去除上一步所有错误点
    if is_print == True:
        print('再次去除一个周期内多个R点后index:', index)
    # print('index:',index)
    # 补充第一个可能的R点
    sum = 0.0
    for i in range(len(index)):
        sum += math.fabs(sig[index[i]])
    mean = sum * 1.0 / len(index)

    dis12 = (int)(mean_dis * 1.2)
    if index[0] > dis12:
        if pos_flag == True:
            pos = np.argmax(sig[0:mean_dis])
        else:
            pos = np.argmin(sig[0:mean_dis])
        index = np.insert(index, 0, pos)
    # 补充最后一个可能的R点
    dis09 = (int)(mean_dis * 0.9)
    if length - index[-1] > dis09:
        if pos_flag == True:
            pos = np.argmax(sig[index[-1] + dis09:]) + index[-1] + dis09
        else:
            pos = np.argmin(sig[index[-1] + dis09:]) + index[-1] + dis09
        if math.fabs(sig[pos]) > mean * 0.4:
            index = np.insert(index, 0, pos)
    if is_print == True:
        print('补充最后可能漏掉的点之后index:', index)

    return index


def detec_qrs(sig, M=True, is_print=False, fs=1000):
    pos_peak = []
    neg_peak = []
    if M == True:
        peroid_len = (int)(452 * fs / 1000)
    else:
        peroid_len = (int)(340 * fs / 1000)
    peroid = (int)(len(sig) / peroid_len) + 1
    for i in range(peroid):
        start = peroid_len * i
        if start == 0:
            start += 1
        end = peroid_len * (i + 1)
        if end >= len(sig):
            end = len(sig) - 1
        if i != 0:
            start -= (int)(50 * fs / 1000)
        max_index = np.argmax(sig[start:end]) + start
        min_index = np.argmin(sig[start:end]) + start
        if sig[max_index] >= sig[max_index - 1] and sig[max_index] >= sig[max_index + 1]:
            pos_peak.append([max_index, sig[max_index]])
        if sig[min_index] <= sig[min_index - 1] and sig[min_index] <= sig[min_index + 1]:
            neg_peak.append([min_index, sig[min_index]])
    pos_peak2 = [ele[0] for ele in pos_peak]
    if M == True:
        pos_peak2 = correct_qrs_inds(sig, pos_peak2, is_print, pos_flag=True, fs=fs)
    else:
        pos_peak2 = correct_qrs_F(sig, pos_peak2, is_print, pos_flag=True, fs=fs)

    neg_peak2 = [ele[0] for ele in neg_peak]
    if M == True:
        neg_peak2 = correct_qrs_inds(sig, neg_peak2, is_print, pos_flag=False, fs=fs)
    else:
        neg_peak2 = correct_qrs_F(sig, neg_peak2, is_print, pos_flag=False, fs=fs)

    if is_print:
        plot_sig_annotation(sig, pos_peak2)
        plot_sig_annotation(sig, neg_peak2)
    # coff_pos = switch(sig, pos_peak2)
    # coff_neg = switch(sig, neg_peak2)
    coff_pos = switch_119(sig, pos_peak2)
    coff_neg = switch_119(sig, neg_peak2)
    # print(coff_pos,coff_neg)
    # plot_sig_annotation(sig,pos_peak2)
    # plot_sig_annotation(sig,neg_peak2)

    if coff_pos > coff_neg:
        return pos_peak2
    else:
        return neg_peak2


def detec_qrs_1218(sig, M=True, is_print=False, fs=1000):
    pos_peak = []
    neg_peak = []
    if M == True:
        peroid_len = (int)(452 * fs / 1000)
    else:
        peroid_len = (int)(340 * fs / 1000)

    lambda_ = 0.01
    pos_len = peroid_len
    start = 0

    while start != len(sig) - 1:
        if start == 0:
            start += 1
        end = start + pos_len
        if end >= len(sig):
            end = len(sig) - 1
        if start != 1:
            start -= (int)(50 * fs / 1000)
        max_index = np.argmax(sig[start:end]) + start
        pos_peak.append(max_index)
        if len(pos_peak) > 1:
            mean = np.mean(np.diff(pos_peak))
            pos_len = int((1 - lambda_) * pos_len + lambda_ * mean)
        start = end

    neg_len = peroid_len
    start = 0
    while start != len(sig) - 1:
        if start == 0:
            start += 1
        end = start + neg_len
        if end >= len(sig):
            end = len(sig) - 1
        if start != 1:
            start -= (int)(50 * fs / 1000)
        min_index = np.argmin(sig[start:end]) + start
        neg_peak.append(min_index)
        if len(neg_peak) > 1:
            mean = np.mean(np.diff(neg_peak))
            neg_len += int((1 - lambda_) * neg_len + lambda_ * mean)
        start = end

    pos_peak2 = pos_peak
    if M == True:
        pos_peak2 = correct_qrs_inds(sig, pos_peak2, is_print, pos_flag=True, fs=fs)
    else:
        pos_peak2 = correct_qrs_F(sig, pos_peak2, is_print, pos_flag=True, fs=fs)

    neg_peak2 = neg_peak
    if M == True:
        neg_peak2 = correct_qrs_inds(sig, neg_peak2, is_print, pos_flag=False, fs=fs)
    else:
        neg_peak2 = correct_qrs_F(sig, neg_peak2, is_print, pos_flag=False, fs=fs)

    if is_print:
        plot_sig_annotation(sig, pos_peak2)
        plot_sig_annotation(sig, neg_peak2)
    # coff_pos = switch(sig, pos_peak2)
    # coff_neg = switch(sig, neg_peak2)
    coff_pos = switch_119(sig, pos_peak2)
    coff_neg = switch_119(sig, neg_peak2)

    if coff_pos > coff_neg:
        return pos_peak2
    else:
        return neg_peak2


from sklearn.decomposition import FastICA


def get_F_ica(ica_mat, length, is_print):
    # ICA（独立成分分析）：可以分析出混合信号中的独立成分，可以作为去噪手段
    # 将传入的多路信号进行ICA分析，得到三个成分，要从中选取能代表胎儿心电那一路
    ica_mat = np.mat(ica_mat)
    ica = FastICA(n_components=3)
    ica_res = ica.fit_transform(ica_mat.T)
    p1 = filt(ica_res[:, 0])
    p2 = filt(ica_res[:, 1])
    p3 = filt(ica_res[:, 2])

    ic_len_flag = [1, 1, 1]  # 初步qrs检测长度是否大于一定值
    inds1 = detec_qrs(p1, M=False, is_print=False)  # 检测心电R峰，参数设置为胎儿
    if len(inds1) < length * 1.2 / 1000 or len(inds1) > length * 3.4 / 1000:  # R峰长度太多太少都不行
        ic_len_flag[0] = 0
    inds2 = detec_qrs(p2, M=False, is_print=False)
    if len(inds2) < length * 1.2 / 1000 or len(inds2) > length * 3.4 / 1000:
        ic_len_flag[1] = 0
    inds3 = detec_qrs(p3, M=False, is_print=False)
    if len(inds3) < length * 1.2 / 1000 or len(inds3) > length * 3.4 / 1000:
        ic_len_flag[2] = 0
    index_list = [inds1, inds2, inds3]

    coeff1 = switchICA(p1, inds1)
    coeff2 = switchICA(p2, inds2)
    coeff3 = switchICA(p3, inds3)
    coff_list = [coeff1, coeff2, coeff3]

    for i in range(3):
        if ic_len_flag[i] == 0:
            coff_list[i] = 0
    # print('coff_list:', coff_list)

    F_index = coff_list.index(max(coff_list))
    FECG_from_ICA = [p1, p2, p3][F_index]
    list1 = index_list[F_index]

    if is_print == True:
        plt.subplot(411).plot(p1)
        plt.title('从ICA结果中选取代表胎儿的信号')
        plt.subplot(412).plot(p2)
        plt.subplot(413).plot(p3)
        plt.subplot(414).plot(filt(FECG_from_ICA), 'r')

        plt.show()
    # detec_F_2(FECG_from_ICA,list1)
    return FECG_from_ICA, list1


def plot_multi_sig_annotation(sig, annList):
    length = len(sig)
    for j in range(length):
        plt.subplot(length, 1, j + 1)
        annArr = np.array([None] * len(sig[j])).T
        for i in range(len(annList)):
            annArr[annList[i]] = sig[j][annList[i]]
        plt.plot(sig[j])
        plt.plot(annArr, 'r+')
    plt.show()


def cancle_MECG_SVD(adECG, mqrs, fs=1000):
    sig = adECG.copy()
    sig = list(sig)
    RR_dist = np.diff(mqrs)
    RR_dist = np.sort(RR_dist)
    mean_RR = np.mean(RR_dist[1:-1])
    before = int(0.2 * mean_RR)
    after = int(0.6 * mean_RR)

    start = 0
    end = len(sig)
    length = len(sig)
    if mqrs[0] < before:
        temp = sig[0]
        for i in range(before - mqrs[0]):
            sig.insert(0, temp)
        start = before - mqrs[0]
    if mqrs[-1] + after > length:
        temp = sig[-1]
        for i in range(after - (length - mqrs[-1])):
            sig.insert(len(sig), temp)
        end = start + length
    # plt.subplot(211)
    # plt.plot(adECG)
    # plt.subplot(212)
    # plt.plot(sig)
    # plt.show()

    A = np.zeros(shape=(before + after, len(mqrs)))

    win = np.ones(before + after)  # 梯形窗
    for i in range(int(30 * fs / 1000)):
        win[i] = 0.033 * i
        win[before + after - 1 - i] = 0.033 * i
    for i in range(len(mqrs)):
        mqrs[i] += start
    # plt.plot(win)
    # plt.show()

    for i in range(len(mqrs)):
        index = mqrs[i]
        A[:, i] = np.multiply(sig[index - before:index + after], win)
    # plt.figure()
    # for i in range(A.shape[1]):
    #     plt.plot(A[:,i])
    # plt.show()
    u, s, v = np.linalg.svd(A, full_matrices=False)
    s = np.diag(s)
    for i in range(1, s.shape[0]):
        s[i, i] = 0
    A_rec = np.dot(np.dot(u, s), v)
    test = np.zeros(shape=(len(sig),))
    for i in range(len(mqrs)):
        index = mqrs[i]
        test[index - before:index + after] = A_rec[:, i]
    # plt.plot(sig)
    # plt.plot(test)
    # plt.subplot(312)
    # plt.plot(sig-test)
    # plt.subplot(313)
    # plt.plot((sig-test)[start:end])
    # plt.show()
    for i in range(len(mqrs)):
        index = mqrs[i]
        sig[index - before:index + after] -= A_rec[:, i]

    return sig[start:end]


def get_peak_valley_ratio(sig, index):
    s = 0
    for i in range(len(index)):
        now_index = index[i]
        if now_index - 180 < 0:
            mean = np.mean(np.abs(sig[now_index + 25:now_index + 180]))
        elif now_index + 180 > len(sig):
            mean = np.mean(np.abs(sig[now_index - 180:now_index - 25]))
        else:
            mean = np.mean(np.abs(sig[now_index + 25:now_index + 180]) + np.abs(sig[now_index - 180:now_index - 25]))
        s = s + abs(sig[now_index]) / mean

        return s / len(index)


# 获取一个序列中方差最小的num个序列
def find_max_minVar(index, num):
    dis = np.diff(index)
    # print(dis)
    min_var = np.var(dis)
    min_var_num = len(dis)
    start_pos = 0
    min_num = num
    for now_num in range(min_num, len(dis)):
        for iteration in range(len(dis) - now_num):
            now_var = np.var(dis[iteration:iteration + now_num])
            # print(min_var,now_var,now_num)
            if now_var < min_var:
                min_var = now_var
                min_var_num = now_num
                start_pos = iteration
    start = start_pos
    end = start + min_var_num
    return start, end


# crtl+z 到这儿就好啦
def detec_F_2(Sig, index, num):
    sig = Sig.copy()
    # plot_sig_annotation(check_polarity(Sig, F=True), index, color='g', title='最初片段')
    start, end = find_max_minVar(index, num)
    # plot_sig_annotation(check_polarity(Sig, F=True), index[start:end], color='g', title='粗检测良好片段')
    pos_num = 0
    neg_num = 0
    pos_flag = False
    for i in range(start, end):
        if sig[index[i]] > 0:
            pos_num += 1
        else:
            neg_num += 1
    if pos_num > neg_num:
        pos_flag = True
    # plot_sig_annotation(sig,index[start:end])
    now_index = list(index[start:end])

    mean_dis = int(np.mean(np.diff(now_index)))
    side_dis = int(mean_dis / 3)

    mean_height = 0
    # 前向推测
    while now_index[0] - mean_dis - side_dis >= 0:
        if mean_dis < 250: break

        mean_height_list = []
        for i in range(len(now_index)):
            mean_height_list.append(abs(Sig[now_index[i]]))
        mean_height = np.mean(mean_height_list)

        # win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 5))  11 4修改
        win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 10))
        pred = now_index[0] - mean_dis
        # print(pred)
        sig[pred - side_dis:pred + side_dis] = np.multiply(sig[pred - side_dis:pred + side_dis], win)
        if pos_flag:
            precise_loc = np.argmax(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
        else:
            precise_loc = np.argmin(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
        if precise_loc < 0: break

        # win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 25))
        # for i in range(len(win)):
        #     if win[i] < 0.4:
        #         win[i] = 0.4
        # sig[precise_loc - side_dis:precise_loc + side_dis] = np.multiply(
        #     sig[precise_loc - side_dis:precise_loc + side_dis], win)

        try:
            if abs(Sig[precise_loc]) < 0.3 * mean_height:
                win = signalTool.windows.gaussian(side_dis * 4, std=int(side_dis * 2))
                sig[pred - side_dis * 2:pred + side_dis * 2] = np.multiply(sig[pred - side_dis * 2:pred + side_dis * 2],
                                                                           win)
                if pos_flag:
                    precise_loc = np.argmax(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2
                else:
                    precise_loc = np.argmin(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2

        except:
            pass

        now_index.insert(0, precise_loc)
        mean_dis = int(np.mean(np.diff(now_index[0:5])))
    # 补充第一个
    end = now_index[0] - mean_dis + side_dis
    # print('now_index:',now_index)
    # print(now_index[0],mean_dis,side_dis)
    if 0 < end < 2 * side_dis:
        win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 5))
        win = win[2 * side_dis - end:]
        # print(end,len(sig[0:end]),len(win))
        sig[0:end] = np.multiply(sig[0:end], win)
        if pos_flag:
            precise_loc = np.argmax(sig[0:end])
        else:
            precise_loc = np.argmin(sig[0:end])
        if abs(sig[precise_loc]) > 0.3 * mean_height:
            now_index.insert(0, precise_loc)

    # 后向推测
    while now_index[-1] + mean_dis + side_dis <= len(sig):
        if mean_dis < 250: break

        mean_height = []
        for i in range(len(now_index)):
            mean_height.append(abs(Sig[now_index[i]]))
        mean_height = np.mean(mean_height)

        win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 5))
        # win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 10))
        pred = now_index[-1] + mean_dis
        # print(pred)
        sig[pred - side_dis:pred + side_dis] = np.multiply(sig[pred - side_dis:pred + side_dis], win)
        if pos_flag:
            precise_loc = np.argmax(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
        else:
            precise_loc = np.argmin(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
        # win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis /25))
        # for i in range(len(win)):
        #     if win[i] < 0.4:
        #         win[i] = 0.4
        # sig[precise_loc - side_dis:precise_loc + side_dis] = np.multiply(
        #     sig[precise_loc - side_dis:precise_loc + side_dis], win)

        try:
            if abs(Sig[precise_loc]) < 0.3 * mean_height:
                win = signalTool.windows.gaussian(side_dis * 4, std=int(side_dis * 2))
                sig[pred - side_dis * 2:pred + side_dis * 2] = np.multiply(sig[pred - side_dis * 2:pred + side_dis * 2],
                                                                           win)
                if pos_flag:
                    precise_loc = np.argmax(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2
                else:
                    precise_loc = np.argmin(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2

        except:
            pass

        now_index.insert(len(now_index), precise_loc)
        mean_dis = int(np.mean(np.diff(now_index[-5:])))

    # 补充第一个可能的R点
    sum = 0.0
    for i in range(len(now_index)):
        sum += math.fabs(sig[now_index[i]])
    mean = sum * 1.0 / len(now_index)

    if now_index[0] > mean_dis * 0.8:
        if now_index[0] - 280 > 0:
            if pos_flag == True:
                pos = np.argmax(now_index[0:now_index[0] - 280])
            else:
                pos = np.argmin(sig[0:now_index[0] - 280])
            if math.fabs(sig[pos]) >= 0.3 * mean:
                now_index.insert(0, pos)

    dis08 = int(0.8 * mean_dis)
    if len(sig) - now_index[-1] > dis08:
        if pos_flag == True:
            pos = np.argmax(sig[now_index[-1] + dis08:]) + now_index[-1] + dis08
        else:
            pos = np.argmin(sig[now_index[-1] + dis08:]) + now_index[-1] + dis08
        if math.fabs(sig[pos]) > mean * 0.3:
            now_index.insert(len(now_index), pos)

    # plot_sig_annotation(check_polarity(Sig, F=True), now_index, title='前后向预测检测结果', color='g')
    # plt.subplot(211).plot(Sig)
    # plt.subplot(212).plot(sig)
    # plt.show()

    return now_index


def detec_F_2_gravity(Sig, index, num, print=False):
    sig = Sig.copy()

    start, end = find_max_minVar(index, num)
    if print:
        plot_sig_annotation(check_polarity(Sig, F=True), index, color='g', title='最初片段')
        plot_sig_annotation(check_polarity(Sig, F=True), index[start:end], color='g', title='粗检测良好片段')
    pos_num = 0
    neg_num = 0
    pos_flag = False
    for i in range(start, end):
        if sig[index[i]] > 0:
            pos_num += 1
        else:
            neg_num += 1
    if pos_num > neg_num:
        pos_flag = True
    # plot_sig_annotation(sig,index[start:end])
    now_index = list(index[start:end])
    mean_dis = int(np.mean(np.diff(now_index)))
    side_dis = int(mean_dis / 3)

    mean_height = 0
    # 前向推测
    while now_index[0] - mean_dis - side_dis >= 0:
        if mean_dis < 250: break

        mean_height_list = []
        for i in range(len(now_index)):
            mean_height_list.append(abs(Sig[now_index[i]]))
        mean_height = np.mean(mean_height_list)

        # win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 5))  11 4修改

        win = []
        M = 1 + abs(1 - (now_index[1] - now_index[0]) / mean_dis)
        for i in range(mean_dis * 2):
            win.append(M / (M + ((i - mean_dis) / (mean_dis / 8)) * ((i - mean_dis) / (mean_dis / 8))))
        win = win[mean_dis - side_dis:mean_dis + side_dis]
        pred = now_index[0] - mean_dis
        # print(pred)
        sig[pred - side_dis:pred + side_dis] = np.multiply(sig[pred - side_dis:pred + side_dis], win)
        if pos_flag:
            precise_loc = np.argmax(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
        else:
            precise_loc = np.argmin(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
        if precise_loc < 0: break

        # win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 25))
        # for i in range(len(win)):
        #     if win[i] < 0.4:
        #         win[i] = 0.4
        # sig[precise_loc - side_dis:precise_loc + side_dis] = np.multiply(
        #     sig[precise_loc - side_dis:precise_loc + side_dis], win)

        try:
            if abs(Sig[precise_loc]) < 0.28 * mean_height:
                # win = signalTool.windows.gaussian(side_dis * 4, std=int(side_dis * 2))
                win = []
                M = 1 + abs(1 - (now_index[1] - now_index[0]) / mean_dis)
                for i in range(mean_dis * 4):
                    win.append(M / (M + ((i - mean_dis * 2) / (mean_dis * 2 / 2.5)) * (
                            (i - mean_dis * 2) / (mean_dis * 2 / 2.5))))
                win = win[mean_dis * 2 - side_dis * 2:mean_dis * 2 + side_dis * 2]
                sig[pred - side_dis * 2:pred + side_dis * 2] = np.multiply(sig[pred - side_dis * 2:pred + side_dis * 2],
                                                                           win)
                if pos_flag:
                    precise_loc = np.argmax(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2
                else:
                    precise_loc = np.argmin(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2

        except:
            pass

        now_index.insert(0, precise_loc)
        mean_dis = int(np.mean(np.diff(now_index[0:5])))
    # 补充第一个
    end = now_index[0] - mean_dis + side_dis
    # print('now_index:',now_index)
    # print(now_index[0],mean_dis,side_dis)
    if 0 < end < 2 * side_dis:
        win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 5))
        win = win[2 * side_dis - end:]
        # print(end,len(sig[0:end]),len(win))
        sig[0:end] = np.multiply(sig[0:end], win)
        if pos_flag:
            precise_loc = np.argmax(sig[0:end])
        else:
            precise_loc = np.argmin(sig[0:end])
        if abs(sig[precise_loc]) > 0.3 * mean_height:
            now_index.insert(0, precise_loc)

    # 后向推测
    while now_index[-1] + mean_dis + side_dis <= len(sig):
        if mean_dis < 250: break

        mean_height = []
        for i in range(len(now_index)):
            mean_height.append(abs(Sig[now_index[i]]))
        mean_height = np.mean(mean_height)

        # win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 5))
        win = []
        M = 1 + abs(1 - (now_index[-1] - now_index[-2]) / mean_dis)
        for i in range(mean_dis * 2):
            win.append(M / (M + ((i - mean_dis) / (mean_dis / 4)) * ((i - mean_dis) / (mean_dis / 4))))
        win = win[mean_dis - side_dis:mean_dis + side_dis]

        pred = now_index[-1] + mean_dis
        # print(pred)
        sig[pred - side_dis:pred + side_dis] = np.multiply(sig[pred - side_dis:pred + side_dis], win)
        if pos_flag:
            precise_loc = np.argmax(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
        else:
            precise_loc = np.argmin(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
        # win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis /25))
        # for i in range(len(win)):
        #     if win[i] < 0.4:
        #         win[i] = 0.4
        # sig[precise_loc - side_dis:precise_loc + side_dis] = np.multiply(
        #     sig[precise_loc - side_dis:precise_loc + side_dis], win)

        try:
            if abs(Sig[precise_loc]) < 0.28 * mean_height:
                # win = signalTool.windows.gaussian(side_dis * 4, std=int(side_dis * 2))
                win = []
                M = 1 + abs(1 - (now_index[-1] - now_index[-2]) / mean_dis)
                for i in range(mean_dis * 4):
                    win.append(
                        M / (M + ((i - mean_dis * 2) / (mean_dis * 2 / 2.5)) * (
                                (i - mean_dis * 2) / (mean_dis * 2 / 2.5))))
                win = win[mean_dis * 2 - side_dis * 2:mean_dis * 2 + side_dis * 2]
                sig[pred - side_dis * 2:pred + side_dis * 2] = np.multiply(sig[pred - side_dis * 2:pred + side_dis * 2],
                                                                           win)
                if pos_flag:
                    precise_loc = np.argmax(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2
                else:
                    precise_loc = np.argmin(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2

        except:
            pass

        now_index.insert(len(now_index), precise_loc)
        mean_dis = int(np.mean(np.diff(now_index[-5:])))

    # 补充第一个可能的R点
    sum = 0.0
    for i in range(len(now_index)):
        sum += math.fabs(sig[now_index[i]])
    mean = sum * 1.0 / len(now_index)

    if now_index[0] > mean_dis * 0.8:
        if now_index[0] - 280 > 0:
            if pos_flag == True:
                pos = np.argmax(now_index[0:now_index[0] - 280])
            else:
                pos = np.argmin(sig[0:now_index[0] - 280])
            if math.fabs(sig[pos]) >= 0.3 * mean:
                now_index.insert(0, pos)

    dis08 = int(0.8 * mean_dis)
    if len(sig) - now_index[-1] > dis08:
        if pos_flag == True:
            pos = np.argmax(sig[now_index[-1] + dis08:]) + now_index[-1] + dis08
        else:
            pos = np.argmin(sig[now_index[-1] + dis08:]) + now_index[-1] + dis08
        if math.fabs(sig[pos]) > mean * 0.3:
            now_index.insert(len(now_index), pos)

    # plot_sig_annotation(check_polarity(Sig, F=True), now_index, title='前后向预测检测结果', color='g')
    # plt.subplot(211).plot(Sig)
    # plt.subplot(212).plot(sig)
    # plt.show()

    return now_index


def detec_F_2_ES(Sig, index, num):
    sig = Sig.copy()
    # plot_sig_annotation(check_polarity(Sig, F=True), index, color='g', title='最初片段')
    start, end = find_max_minVar(index, num)
    # plot_sig_annotation(check_polarity(Sig, F=True), index[start:end], color='g', title='粗检测良好片段')
    pos_num = 0
    neg_num = 0
    pos_flag = False
    for i in range(start, end):
        if sig[index[i]] > 0:
            pos_num += 1
        else:
            neg_num += 1
    if pos_num > neg_num:
        pos_flag = True
    # plot_sig_annotation(sig,index[start:end])
    now_index = list(index[start:end])
    temp = np.mean(np.diff(now_index)[1:])  # 存储历史预测值，初始值用前几个的均值代替
    alpha = 0.1

    mean_dis = int(np.mean(np.diff(now_index)))
    side_dis = int(mean_dis / 3)
    # print(np.diff(now_index))

    mean_height = 0
    precise_loc = 0
    pred = 0
    # 前向推测
    while now_index[0] > 0:
        if mean_dis < 250: break

        mean_height_list = []
        for i in range(len(now_index)):
            mean_height_list.append(abs(Sig[now_index[i]]))
        mean_height = np.mean(mean_height_list)
        try:
            # win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 5))
            win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 10))
            pred = int(now_index[0] - (alpha * (now_index[1] - now_index[0]) + (1 - alpha) * temp))  # 指数平滑预测
            # print(pred)
            temp = alpha * np.diff(now_index)[0] + (1 - alpha) * temp  # 更新预测值
            # print(temp)
            sig[pred - side_dis:pred + side_dis] = np.multiply(sig[pred - side_dis:pred + side_dis], win)
            if pos_flag:
                precise_loc = np.argmax(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
            else:
                precise_loc = np.argmin(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
            if precise_loc < 0: break
        except:
            pass

        try:
            if abs(Sig[precise_loc]) < 0.3 * mean_height:
                win = signalTool.windows.gaussian(side_dis * 4, std=int(side_dis * 2))
                sig[pred - side_dis * 2:pred + side_dis * 2] = np.multiply(sig[pred - side_dis * 2:pred + side_dis * 2],
                                                                           win)
                if pos_flag:
                    precise_loc = np.argmax(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2
                else:
                    precise_loc = np.argmin(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2

        except:
            pass

        now_index.insert(0, precise_loc)
        mean_dis = int(np.mean(np.diff(now_index[0:5])))
    # 补充第一个
    end = now_index[0] - mean_dis + side_dis
    # print('now_index:',now_index)
    # print(now_index[0],mean_dis,side_dis)
    if 0 < end < 2 * side_dis:
        win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 5))
        win = win[2 * side_dis - end:]
        # print(end,len(sig[0:end]),len(win))
        sig[0:end] = np.multiply(sig[0:end], win)
        if pos_flag:
            precise_loc = np.argmax(sig[0:end])
        else:
            precise_loc = np.argmin(sig[0:end])
        if abs(sig[precise_loc]) > 0.3 * mean_height:
            now_index.insert(0, precise_loc)

    # 后向推测
    temp = np.mean(np.diff(now_index[-1 * num:-1]))  # 存储历史预测值，初始值用前几个的均值代替
    while now_index[-1] + mean_dis + side_dis <= len(sig):
        if mean_dis < 250: break

        mean_height = []
        for i in range(len(now_index)):
            mean_height.append(abs(Sig[now_index[i]]))
        mean_height = np.mean(mean_height)

        try:
            win = signalTool.windows.gaussian(side_dis * 2, std=int(mean_dis / 5))
            pred = int(now_index[-1] + (alpha * (now_index[-1] - now_index[-2]) + (1 - alpha) * temp))
            # print(pred)
            temp = alpha * (now_index[-1] - now_index[-2]) + (1 - alpha) * temp
            sig[pred - side_dis:pred + side_dis] = np.multiply(sig[pred - side_dis:pred + side_dis], win)
            if pos_flag:
                precise_loc = np.argmax(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
            else:
                precise_loc = np.argmin(sig[pred - side_dis:pred + side_dis]) + pred - side_dis
        except:
            pass

        try:
            if abs(Sig[precise_loc]) < 0.3 * mean_height:
                win = signalTool.windows.gaussian(side_dis * 4, std=int(side_dis * 2))
                sig[pred - side_dis * 2:pred + side_dis * 2] = np.multiply(sig[pred - side_dis * 2:pred + side_dis * 2],
                                                                           win)
                if pos_flag:
                    precise_loc = np.argmax(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2
                else:
                    precise_loc = np.argmin(sig[pred - side_dis * 2:pred + side_dis * 2]) + pred - side_dis * 2

        except:
            pass

        now_index.insert(len(now_index), precise_loc)
        mean_dis = int(np.mean(np.diff(now_index[-5:])))

    # 补充第一个可能的R点
    sum = 0.0
    for i in range(len(now_index)):
        sum += math.fabs(sig[now_index[i]])
    mean = sum * 1.0 / len(now_index)

    if now_index[0] > mean_dis * 0.8:
        if now_index[0] - 280 > 0:
            if pos_flag == True:
                pos = np.argmax(now_index[0:now_index[0] - 280])
            else:
                pos = np.argmin(sig[0:now_index[0] - 280])
            if math.fabs(sig[pos]) >= 0.3 * mean:
                now_index.insert(0, pos)

    dis08 = int(0.8 * mean_dis)
    if len(sig) - now_index[-1] > dis08:
        if pos_flag == True:
            pos = np.argmax(sig[now_index[-1] + dis08:]) + now_index[-1] + dis08
        else:
            pos = np.argmin(sig[now_index[-1] + dis08:]) + now_index[-1] + dis08
        if math.fabs(sig[pos]) > mean * 0.3:
            now_index.insert(len(now_index), pos)

    # plot_sig_annotation(check_polarity(Sig, F=True), now_index, title='前后向预测检测结果', color='g')
    # plt.subplot(211).plot(Sig)
    # plt.subplot(212).plot(sig)
    # plt.show()

    return now_index


def switch_ica_from_3(p1, p2, p3, length):
    ic_len_flag = [1, 1, 1]  # 初步qrs检测长度是否大于一定值
    inds1 = detec_qrs(p1, M=False, is_print=False)  # 检测心电R峰，参数设置为胎儿
    if len(inds1) < length * 1.2 / 1000 or len(inds1) > length * 3.4 / 1000:  # R峰长度太多太少都不行
        ic_len_flag[0] = 0
    inds2 = detec_qrs(p2, M=False, is_print=False)
    if len(inds2) < length * 1.2 / 1000 or len(inds2) > length * 3.4 / 1000:
        ic_len_flag[1] = 0
    inds3 = detec_qrs(p3, M=False, is_print=False)
    if len(inds3) < length * 1.2 / 1000 or len(inds3) > length * 3.4 / 1000:
        ic_len_flag[2] = 0
    index_list = [inds1, inds2, inds3]

    coeff1 = switchICA(p1, inds1)
    coeff2 = switchICA(p2, inds2)
    coeff3 = switchICA(p3, inds3)
    coff_list = [coeff1, coeff2, coeff3]

    for i in range(3):
        if ic_len_flag[i] == 0:
            coff_list[i] = 0
    # print('coff_list:', coff_list)

    F_index = coff_list.index(max(coff_list))
    FECG_from_ICA = [p1, p2, p3][F_index]
    list1 = index_list[F_index]

    # detec_F_2(FECG_from_ICA,list1)
    return FECG_from_ICA, list1


def sigmoid(x, alpha_: float = 1):
    return 1 / (1 + np.exp(-1 * x / alpha_))


def local_ruili(sig1, alpha=4, tao=10):
    power = np.power
    log = np.log

    sig = sig1
    sig = abs(sig)

    sig = sig / (1.1 * max(sig))
    length = len(sig)
    deri = []

    for i in range(length):
        start = 0 if i - tao < 0 else i - tao
        end = length if i + tao > length else i + tao
        deri.append(sigmoid(log(np.sum(power(sig[start:end], alpha))), alpha_=1.5))
    return np.multiply(sig1, deri)


def get_max_abs(sig):  # 将信号放缩到-1到1
    max_abs = max(abs(max(sig)), abs(min(sig)))
    return sig / max_abs


import seaborn as sns
def local_renyi_en(sig,tao):
    npersig = 30
    noverlap = 29
    f, t, zxx = stft(sig, fs=1000, nperseg=npersig, return_onesided=True, noverlap=noverlap)
    M = np.abs(zxx)
    # #
    # f=f[0:6]
    # t*=1000
    # # M=M[0:6,]
    # M/=np.max(M)
    #
    # cm = plt.get_cmap('jet')
    # plt.pcolormesh(t, f[0:6], M[0:6,])
    # # plt.colorbar()
    # plt.show()
    #
    R = np.zeros(shape=(M.shape[1],))
    power = np.power
    sum = np.sum

    for column in range(len(R)):
        t_start = column - tao if column - tao >= 0 else 0
        t_end = column + tao if column + tao <= len(R) else len(R)
        molecular = sum(power(M[:, t_start:t_end], 1))
        # denominator=power(np.sum(power(M[:,t_start:t_end],2)),2)
        R[column] = molecular
    # fig, ax = plt.subplots()
    # fig.patch.set_alpha(0.)
    # plt.plot(R[0:4500],'black',linewidth=1.5)
    # plt.axis('off')
    # plt.show()
    rec = np.multiply(R[:-1], sig)
    return rec


def local_renyi_en2(sig):
    sig=sig[0:3300]
    npersig = 30
    noverlap = 29
    f, t, zxx = stft(sig, fs=1000, nperseg=npersig, return_onesided=True, noverlap=noverlap)
    M = np.abs(zxx)
    #
    cm=plt.get_cmap('magma')
    plt.pcolormesh(t, f, M)
    plt.colorbar()
    plt.show()
    #
    R = np.zeros(shape=(M.shape[1],))
    power = np.power
    sum = np.sum

    tao = 45
    for column in range(len(R)):
        t_start = column - tao if column - tao >= 0 else 0
        t_end = column + tao if column + tao <= len(R) else len(R)
        molecular = sum(power(M[:, t_start:t_end], 1))
        # denominator=power(np.sum(power(M[:,t_start:t_end],2)),2)
        R[column] = molecular

    rec = np.multiply(R[:-1], sig)
    return rec


def get_kurtosis(sig, index):
    coeff = 0
    tao = 25
    for i in range(len(index)):
        peak = abs(sig[index[i]])
        start = index[i] - tao if index[i] - tao >= 0 else 0
        end = index[i] + tao if index[i] + tao >= len(sig) else len(sig)
        vally = np.mean(np.abs(sig[start:end]))
        coeff += peak / vally
    return coeff / len(index)
