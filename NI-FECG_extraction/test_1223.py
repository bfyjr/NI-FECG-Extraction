import wfdb
import numpy as np
import math
import time
import matplotlib
from matplotlib import pyplot as plt
import warnings
from scipy import io

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.decomposition import NMF
from scipy.signal import stft, istft
from sklearn.decomposition import PCA, FastICA, KernelPCA
from tool822_108 import remove_nan
from tool822_108 import cancle_MECG_HF as cancel_MECG_LMS
from tool822_108 import filt
from tool822_108 import local_ruili
from tool822_108 import get_kurtosis
from tool822_108 import local_renyi_en
from tool822_108 import plot_sig_annotation, plot_multi_sig_annotation
# from Stable822.tool822_108 import switchICA_envelope as switchICA
from tool_14_for_fig import switchICA_envelope as switchICA
from tool822_108 import is_channel_qualified
from tool822_108 import detec_qrs as detec_qrs
from tool822_108 import get_signal_from_to
# from Stable1223.tool_14_for_fig import detec_F_2_gravity as detec_F_2
from tool_14_for_fig import detec_F_2_gravity as detec_F_2
from tool822_108 import check_polarity
import os
# from biosppy.signals import ecg
from wfdb import processing


def get_max_abs(sig):  # 将信号放缩到-1到1
    max_abs = max(abs(max(sig)), abs(min(sig)))
    return sig / max_abs


print('begin:')

all_acc_list = []
all_se_list=[]
all_F1_list = []
all_MAE = 0

tao_list=[]
for i in range(18,19):
    tao_list.append(i)
for iter_number in range(len(tao_list)):
    print('tao=',tao_list[iter_number])
    MAE = 0
    acc_list = []
    se_list=[]
    F1_list = []
    TP_iter_number = 0  # 统计一次循环所有69个文件所有TP
    # file_iter_list = np.arange(1, 76)
    # file_iter_list = np.delete(file_iter_list, 17)
    # file_iter_list = np.insert(file_iter_list, 0, 18)
    for file_iter in range(1,76):
        if file_iter in [33, 38, 52, 54, 71, 74]: continue
        if file_iter < 10:
            file_iter = '0' + str(file_iter)
        else:
            file_iter = str(file_iter)

        path = 'set_a\\a' + file_iter
        sampfrom = 500
        sampto = 59500
        signal, fields = wfdb.rdsamp(path, channels=[0, 1, 2, 3], sampfrom=sampfrom, sampto=sampto)

        # signal=io.loadmat('Xf.mat').get('Xd')
        # print(signal.shape)
        #
        # annotation = wfdb.rdann(path, 'fqrs', sampfrom=10000, sampto=20000)
        # annList = annotation.__dict__.get('sample')
        # print('注释：',annList-10000)

        aa = signal[:, 0]
        bb = signal[:, 1]
        cc = signal[:, 2]
        dd = signal[:, 3]

        # lines=plt.subplot(411).plot(aa)
        # plt.title('预处理前的信号')
        # plt.subplot(412).plot(bb)
        # plt.subplot(413).plot(cc)
        # plt.subplot(414).plot(dd)
        # plt.show()
        #
        # 去除信号采集中的无效值
        aa = remove_nan(aa)
        bb = remove_nan(bb)
        cc = remove_nan(cc)
        dd = remove_nan(dd)
        #
        #
        #
        # # 滤波，低通去除高频噪声，高通去除基线漂移，陷波滤波器去除50hz和60hz的电流噪声
        aa = filt(aa)
        bb = filt(bb)
        cc = filt(cc)
        dd = filt(dd)
        #
        # plt.subplot(411).plot(aa)
        # plt.title('预处理后的信号')
        # plt.subplot(412).plot(bb)
        # plt.subplot(413).plot(cc)
        # plt.subplot(414).plot(dd)
        # plt.show()
        #
        #
        # # 将信号放缩到-1到1
        aa = get_max_abs(aa)
        bb = get_max_abs(bb)
        cc = get_max_abs(cc)
        dd = get_max_abs(dd)

        # 检测四组信号的质量是否良好，取良好的几个做后续分析，测试标准是根据心电信号的特点，

        aECG_flag = [0, 0, 0, 0]
        if is_channel_qualified(aa, len(aa)):
            aECG_flag[0] = 1
        if is_channel_qualified(bb, len(bb)):
            aECG_flag[1] = 1
        if is_channel_qualified(cc, len(cc)):
            aECG_flag[2] = 1
        if is_channel_qualified(dd, len(dd)):
            aECG_flag[3] = 1

        #  最后用来计算的指标
        TP = 0
        FN = 0
        FP = 0

        total_acc = 0
        total_se=0

        sig_to_list = get_signal_from_to(aa, bb, cc, dd, aECG_flag)
        sig_to_list.append(len(aa) - 1)

        # 保存每个记录的检测结果和参考结果
        iter_detected_list = []
        iter_ann_list = []

        # if is_print:
        #     plot_sig_annotation(aa,sig_to_list)
        print('----------------------------------------------------------------------------------------------')
        is_print = False  # 是否显示图
        for iteration in range(6):
            # print('----------------------------------------------------------------------------------------------')
            if iteration == 0:
                sig_from = 0
            else:
                sig_from = sig_to_list[iteration - 1]
            start = time.clock()
            print('第', path[-3:], '个文件第：', iteration, '次循环')
            sig_to = sig_to_list[iteration]
            length = sig_to - sig_from
            # print(sig_from, '---', sig_to)
            # 读取胎儿R峰参考点
            # annList=0
            annotation = wfdb.rdann(path, 'fqrs', sampfrom=sig_from + sampfrom, sampto=sig_to + sampfrom)
            annList = annotation.__dict__.get('sample')
            for i in range(len(annList)):
                iter_ann_list.append(annList[i]-sampfrom)
            annList = annList - sig_from - sampfrom

            fs = 1000  # 取样率

            a = aa[sig_from:sig_to]
            b = bb[sig_from:sig_to]
            c = cc[sig_from:sig_to]
            d = dd[sig_from:sig_to]

            aECG = []
            for i in range(len(aECG_flag)):
                if aECG_flag[i] == 1:
                    aECG.append([a, b, c, d][i])
            # print(aECG_flag)

            if is_print == True:
                plt.subplot(411).plot(a)
                plt.title('预处理后的信号-Main822 -' + str(file_iter) + '-' + str(iteration))
                plt.subplot(412).plot(b)
                plt.subplot(413).plot(c)
                plt.subplot(414).plot(d)
                plt.show()

            component_num = 3  # 信号中的成分数，一般认为包含，母亲，胎儿，噪声三种
            if len(aECG) < 2:  # 4路信号中只有一路合格，后面用到的独立成分分析方法至少需要2种成分，为了方便，全部分析
                aECG = [a, b, c, d]
            elif len(aECG) == 2:  # 4路中有2路合格，2路信号不能分离三个独立成分，所以信号中的成分设置成两个
                component_num = 2
            sig_mat = np.mat(aECG).T
            # 使用KernelPCA+ICA(核主成分分析)对母体信号进行增强，方便识别R峰位置
            kpca = KernelPCA(n_components=component_num, kernel='poly', degree=4, n_jobs=-1)  # 调库，使用多项式核，开启cpu所有核心加速
            temp = kpca.fit_transform(sig_mat)

            if not is_print:
                for i in range(temp.shape[1]):
                    temp[:, i] = local_renyi_en(temp[:, i],tao=tao_list[iter_number])
            if is_print==True:
                plt.figure()
                for i in range(temp.shape[1]):
                    plt.subplot(2 * temp.shape[1], 1, i * 2 + 1).plot(temp[:, i], 'g')
                    temp[:, i] = local_renyi_en(temp[:, i],tao=tao_list[iter_number])
                    plt.subplot(2 * temp.shape[1], 1, i * 2 + 2).plot(temp[:, i], 'r')
                plt.show()

            if component_num == 2:
                w_init = None
            ica2 = FastICA(n_components=component_num)
            ica2_res = ica2.fit_transform(temp)  # 得到component_num（2或3）组信号，包含增强后的母体信号

            ic1 = get_max_abs(ica2_res[:, 0])
            ic2 = get_max_abs(ica2_res[:, 1])
            if component_num == 3:
                ic3 = get_max_abs(ica2_res[:, 2])
            else:
                ic3 = get_max_abs(np.random.random(size=ica2_res[:, 1].shape))
            ic_list = [ic1, ic2, ic3]
            # print(length)
            min_M_qrs = length * 0.6 / 1000  # 成人心跳在60-110之间
            max_M_qrs = math.ceil(length / 500)  # 这两个都是自定义的阈值
            # 自定义的检测心电信号QRS的方法，识别R峰位置，返回一个位置序列
            qrs1 = detec_qrs(ic1, M=True)  # 上一步KernelICA得到的三个成分，均进行R检测，得到三个R序列
            qrs2 = detec_qrs(ic2, M=True)
            qrs3 = detec_qrs(ic3, M=True)

            # print(qrs1)
            # print(qrs2)
            # print(qrs3)
            # print(np.var(np.diff(qrs1)),np.var(np.diff(qrs2)),np.var(np.diff(qrs3)))

            # 以下这段是从三路信号中选取峰值信号最明显的一路的选择流程
            coff = []
            is_peak_list = []
            qrs_list = [qrs1, qrs2, qrs3]
            envelope_list = []
            # 选取能代表母体信号的信号
            for i in range(len(qrs_list)):
                if len(qrs_list[i]) > max_M_qrs or len(qrs_list[i]) <= min_M_qrs:
                    # plot_sig_annotation(ic_list[i],qrs_list[i])
                    coff.append(0)
                    is_peak_list.append(10000)
                else:
                    [all_num, coff1, envelope] = switchICA(ic_list[i], qrs_list[i], M=True)
                    envelope_list.append(envelope)
                    coff.append(coff1)
                    is_peak_list.append(all_num)
            # print(is_peak_list)
            # print(coff)
            if is_print:
                plt.subplot(311).plot(ic1, 'darkorange', linewidth=0.8)
                # plt.axis('off')
                plt.subplot(312).plot(ic2, 'darkorange', linewidth=0.8)
                # plt.axis('off')
                plt.subplot(313).plot(ic3, 'cornflowerblue', linewidth=0.8)
                # plt.axis('off')
                plt.show()
                plt.subplot(311).plot(envelope_list[0], 'g', linewidth=0.8)
                annArr = np.array([None] * len(envelope_list[0])).T
                for i in range(len(qrs1)):
                    annArr[qrs1[i]] = envelope_list[0][qrs1[i]]
                plt.subplot(311).plot(annArr, 'r+')
                plt.axis('off')

                plt.subplot(312).plot(envelope_list[1], 'g', linewidth=0.8)
                annArr = np.array([None] * len(envelope_list[1])).T
                for i in range(len(qrs2)):
                    annArr[qrs2[i]] = envelope_list[1][qrs2[i]]
                plt.subplot(312).plot(annArr, 'r+')
                plt.axis('off')

                plt.subplot(313).plot(envelope_list[2], 'g', linewidth=0.8)
                annArr = np.array([None] * len(envelope_list[2])).T
                for i in range(len(qrs3)):
                    annArr[qrs3[i]] = envelope_list[2][qrs3[i]]
                plt.subplot(313).plot(annArr, 'r+')
                plt.axis('off')
                plt.show()

            kurtosis_list = []
            for i in range(len(ic_list)):
                kurtosis_list.append(get_kurtosis(ic_list[i], qrs_list[i]))
            # print(kurtosis_list)
            if is_peak_list[0] == is_peak_list[1] and is_peak_list[1] == is_peak_list[2]:
                # M_index = np.argmax(coff)
                M_index = np.argmax(kurtosis_list)
            else:
                if len(set(is_peak_list)) == len(is_peak_list):
                    M_index = np.argmin(is_peak_list)
                else:
                    s = set(is_peak_list)
                    temp_list = []
                    for i in s:
                        if is_peak_list.count(i) > 1 and i != 10000:
                            for j in range(len(is_peak_list)):
                                if is_peak_list[j] == i:
                                    temp_list.append(coff[j])
                                else:
                                    temp_list.append(0)
                    if temp_list.__len__() == 0:
                        M_index = np.argmax(coff)
                    else:
                        if is_peak_list.count(0) == 1:
                            M_index = np.argmin(is_peak_list)
                        else:
                            M_index = np.argmax(temp_list)
            # print('M_index:',M_index)
            sig = ic_list[M_index]
            qrs_index = detec_qrs(sig, M=True, is_print=False)
            # 得到母体信号准确定位：
            # print(qrs_index)
            if is_print == True:
                plt.subplot(411).plot(ic1)
                annArr = np.array([None] * len(ic1)).T
                for i in range(len(qrs1)):
                    annArr[qrs1[i]] = ic1[qrs1[i]]
                plt.plot(annArr,'r+')
                plt.title('使用KernelICA增强的母体信号，从三路中选一路识别母体R峰 使用local renyi')
                plt.subplot(412).plot(ic2)
                annArr = np.array([None] * len(ic2)).T
                for i in range(len(qrs2)):
                    annArr[qrs2[i]] = ic2[qrs2[i]]
                plt.plot(annArr, 'r+')
                plt.subplot(413).plot(ic3)
                annArr = np.array([None] * len(ic3)).T
                for i in range(len(qrs3)):
                    annArr[qrs3[i]] = ic3[qrs3[i]]
                plt.plot(annArr, 'r+')
                plt.subplot(414).plot(sig)
                plt.show()
            if is_print == True:
                plot_sig_annotation(sig, qrs_index)

            # getBeatMat(a,qrs_index)
            # 使用基于最小均方误差的模板去除方法去除母体信号：大概流程：
            # 根据得到的母体R峰将十秒内每一个R峰左边0.25，右边0.45秒信号截取，求和取平均，得到一个母体信号的一个周期的平均模板
            # 将母体心跳周期模板根据论文中方法放缩适应比例大小，从原始信号中直接减去，得到含有噪声胎儿信号
            # plot_multi_sig_annotation([a,b,c,d],qrs_index)

            a1 = cancel_MECG_LMS(a.copy(), qrs_index)
            b1 = cancel_MECG_LMS(b.copy(), qrs_index)
            c1 = cancel_MECG_LMS(c.copy(), qrs_index)
            d1 = cancel_MECG_LMS(d.copy(), qrs_index)


            # io.savemat('fecg.mat',{'fecg':np.mat([a1,b1,c1,d1])})

            # coff_cancel_MECG=[]
            # for i in [a1,b1,c1,d1]:
            #     coff_cancel_MECG.append(switchICA(i,detec_qrs(i,M=False)))
            if  is_print:
                matplotlib.rcParams.update({'font.size': 40})
                plt.subplot(411).plot(a1,'y')
                plt.title('(a):FECG after MECG cancellation of record \'a'+file_iter+'\'in set-A')
                plt.subplot(412).plot(b1, 'gold')
                plt.subplot(413).plot(c1, 'r')
                plt.subplot(414).plot(d1, 'g')
                plt.show()
                plt.close()
                matplotlib.rcParams.update({'font.size': 13})
            if is_print == True:
                plt.subplot(811).plot(a)
                plt.title('原始信号及去除母体之后的信号')
                plt.subplot(812).plot(a1, 'r')
                plt.subplot(813).plot(b)
                plt.subplot(814).plot(b1, 'r')
                plt.subplot(815).plot(c)
                plt.subplot(816).plot(c1, 'r')
                plt.subplot(817).plot(d)
                plt.subplot(818).plot(d1, 'r')
                plt.show()


            # a1 = detec_2(a1, detec_qrs(a1, M=True))
            # b1 = detec_2(b1, detec_qrs(b1, M=True))
            # c1 = detec_2(c1, detec_qrs(c1, M=True))
            # d1 = detec_2(d1, detec_qrs(d1, M=True))
            #
            # if is_print == True:
            #     plt.subplot(811).plot(a)
            #     plt.title('原始信号及去除母体之后的信号2222')
            #     plt.subplot(812).plot(a1, 'r')
            #     plt.subplot(813).plot(b)
            #     plt.subplot(814).plot(b1, 'r')
            #     plt.subplot(815).plot(c)
            #     plt.subplot(816).plot(c1, 'r')
            #     plt.subplot(817).plot(d)
            #     plt.subplot(818).plot(d1, 'r')
            #     plt.show()

            def get_F_ica(ica_mat):
                # ICA（独立成分分析）：可以分析出混合信号中的独立成分，可以作为去噪手段
                # 将传入的多路信号进行ICA分析，得到三个成分，要从中选取能代表胎儿心电那一路
                ica_mat = np.mat(ica_mat)
                ica = FastICA(n_components=3,fun='exp')
                ica_res = ica.fit_transform(ica_mat.T)
                # print(ica.n_iter_,ica.tol)

                p1 = filt(ica_res[:, 0])
                p2 = filt(ica_res[:, 1])
                p3 = filt(ica_res[:, 2])

                p1 = check_polarity(p1, fs=1000, F=True)
                p2 = check_polarity(p2, fs=1000, F=True)
                p3 = check_polarity(p3, fs=1000, F=True)

                ic_len_flag = [1, 1, 1]  # 初步qrs检测长度是否大于一定值
                inds1 = detec_qrs(p1, M=False, is_print=False)  # 检测心电R峰，参数设置为胎儿
                # inds1=detec_F_2(p1,inds1,5)
                if len(inds1) < length * 1.2 / 1000 or len(inds1) > length * 3.4 / 1000:  # R峰长度太多太少都不行
                    ic_len_flag[0] = 0
                inds2 = detec_qrs(p2, M=False, is_print=False)
                # inds2 = detec_F_2(p2, inds2, 5)
                if len(inds2) < length * 1.2 / 1000 or len(inds2) > length * 3.4 / 1000:
                    ic_len_flag[1] = 0
                inds3 = detec_qrs(p3, M=False, is_print=False)
                # inds3 = detec_F_2(p3, inds3, 5)
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
                    matplotlib.rcParams.update({'font.size': 40})
                    plt.subplot(311).plot(p1,'y')
                    plt.title('(b):ICA Output Of FECG After MECG Cancellation ')
                    plt.subplot(312).plot(p2,'gold')
                    plt.subplot(313).plot(p3,'r')
                    plt.show()
                    matplotlib.rcParams.update({'font.size': 13})

                    plt.subplot(411).plot(p1)
                    annArr = np.array([None] * len(p1)).T
                    for i in range(len(inds1)):
                        annArr[inds1[i]] = p1[inds1[i]]
                    plt.subplot(411).plot(annArr, 'r+')
                    plt.title(' cancle_MECG_HF_2 从ICA结果中选取代表胎儿的信号;coff_list:' + str(coff_list))

                    plt.subplot(412).plot(p2)
                    annArr = np.array([None] * len(p2)).T
                    for i in range(len(inds2)):
                        annArr[inds2[i]] = p2[inds2[i]]
                    plt.subplot(412).plot(annArr, 'r+')

                    plt.subplot(413).plot(p3)
                    annArr = np.array([None] * len(p3)).T
                    for i in range(len(inds3)):
                        annArr[inds3[i]] = p3[inds3[i]]
                    plt.subplot(413).plot(annArr, 'r+')
                    plt.subplot(414).plot(filt(FECG_from_ICA), 'r')

                    plt.show()
                # detec_F_2(FECG_from_ICA,list1)
                return FECG_from_ICA, list1


            # 把四路含噪声的胎儿信号排列组合一下，分别提取独立成分中的胎儿信号，得到5路信号，放在F_ic_list中，对应R峰序列放在F_index_list
            combine = [[a1, b1, c1, d1], [b1, c1, d1], [a1, c1, d1], [a1, b1, d1], [a1, b1, c1]]
            # combine = [[b1, c1, d1], [a1, c1, d1], [a1, b1, d1], [a1, b1, c1]]#  219修改
            # if component_num==3:
            #     max_pos=np.argmax(coff_cancel_MECG)
            #     combine.pop(max_pos+1)

            if component_num == 2:
                combine2 = [[a1, b1, c1, d1]]
                for n in range(len(aECG_flag)):
                    if aECG_flag[n] == 0:
                        combine2.append(combine[n + 1])
                combine = combine2
            F_ic_list = []
            if is_print == True:
                plt.figure()
                plt.title('每次窗口得到的ica结果')
            F_index = []
            F_sig = []
            rec_list=[]
            for j in range(len(combine)):
                sig, index = get_F_ica(combine[j])
                F_sig.append(sig)
                index ,rec= detec_F_2(sig, index, 5, is_print=False)
                index, rec = detec_F_2(rec, index, 8, is_print=False)
                # index, rec = detec_F_2(rec, index, 8, is_print=False)
                # xqrs0 = processing.XQRS(sig=sig*1000, fs=1000)
                # xqrs0.detect()
                # index=xqrs0.qrs_inds

                rec_list.append(rec)
                # index=ecg.hamilton_segmenter(sig, sampling_rate=1000)
                # F_index.append(index[0])
                F_index.append(index)

            coff_list = []
            for i in range(len(F_index)):
                coff = switchICA(F_sig[i], F_index[i])
                coff_list.append(coff)
                # plot_sig_annotation(F_sig[i],F_index[i],title=coff)

            f_pos = int(np.argmax(coff_list))
            list1 = F_index[f_pos]
            rec_sig=rec_list[f_pos]
            # np.save('FECG_and_fqrs_set-A\\'+file_iter+'-'+str(iteration)+'.npy',[rec_sig,list1,qrs_index],allow_pickle=True)
            if  is_print==True:
                plot_sig_annotation(rec_sig, list1, title='cancle_MECG_HF_2 预测')
                plot_sig_annotation(rec_sig, annList,'cancle_MECG_HF_2 标注')


            if is_print == True:
                # plt.figure()
                for j in range(len(F_sig)):
                    plt.subplot(len(F_sig) + 1, 1, j + 1).plot(F_sig[j])
                    annArr = np.array([None] * len(F_sig[j])).T
                    for i in range(len(F_index[j])):
                        annArr[F_index[j][i]] = F_sig[j][F_index[j][i]]
                    plt.subplot(len(F_sig) + 1, 1, j + 1).plot(annArr, 'r+')
                    if j == 0: plt.title(str(coff_list))
                plt.subplot(len(F_index) + 1, 1, len(F_index) + 1).plot(F_sig[f_pos])
                plt.show()

            for i in range(len(list1)):
                iter_detected_list.append(list1[i]+sig_from)

            corrcet_det_num = 0
            tp = 0
            fn = 0
            fp = 0
            # print(list1)
            # print(annList)
            for i in range(len(list1)):
                for j in range(len(annList)):
                    if math.fabs(list1[i] - annList[j]) <= 50:
                        MAE += math.fabs(list1[i] - annList[j])
                        annList[j] = -100
                        corrcet_det_num += 1
                        break
            now_acc = corrcet_det_num * 1.0 / len(annList)
            now_se=corrcet_det_num*1.0/len(list1)
            tp = corrcet_det_num
            fn = len(annList) - corrcet_det_num
            fp = len(list1) - corrcet_det_num

            TP += tp
            FN += fn
            FP += fp
            print('se:',now_se,'      acc:', corrcet_det_num, '/', len(annList), '=', now_acc, '  f1= ', tp * 2 / (tp * 2 + fn + fp))
            total_acc += now_acc
            total_se+=now_se
            # print('探测时间：', time.clock() - start)
            # if is_print == True:
            #     plot_sig_annotation(F_ic_list[F_index],list1)
            sig_from = sig_to


        print('平均se：', total_se / 6,'    平均准确率：', total_acc / 6, '    F1值：', TP * 2 / (TP * 2 + FN + FP))
        TP_iter_number += TP
        acc_list.append(total_acc / 6)
        se_list.append(total_se/6)
        F1_list.append(TP * 2 / (TP * 2 + FN + FP))


        # np.save('ann_detected_1_27\\'+file_iter+'_ann',iter_ann_list)
        # np.save('ann_detected_1_27\\'+file_iter+'_detected',iter_detected_list)
        # np.save('ann_detected_1_27\\'+file_iter+'_FECG',rec_sig)
    print(5, '下', '最后平均se: ', np.mean(se_list),'    最后平均acc: ', np.mean(acc_list), '    最后平均F1: ', np.mean(F1_list), '  MAE:', str(MAE / TP_iter_number))
    all_acc_list.append(np.mean(acc_list))
    all_se_list.append(np.mean(se_list))
    all_F1_list.append(np.mean(F1_list))
    print('nowMAE:', MAE)
    MAE /= TP_iter_number
    all_MAE += MAE
    print('*****************************************************')
print(np.mean(all_acc_list), all_acc_list)
print(np.mean(all_F1_list), all_F1_list)
print(np.mean(all_se_list),all_se_list)
print('MAE:', all_MAE / len(tao_list))
np.save('acc.npy',all_acc_list)
np.save('F1.npy',all_F1_list)
np.save('se.npy',all_se_list)

