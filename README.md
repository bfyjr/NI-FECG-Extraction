Non-invasive Fetal electrocardiogram (NI-FECG) Extraction
===
无创胎儿心电信号提取
---
背景：胎儿心电图（FECG）监测有助于在分娩前分析胎儿的健康状况。 将FECG与从母体腹部收集的非侵入性ECG信号分离是一项艰巨的任务，这是由于FECG和母体ECG（MECG）的重叠，胎儿R峰的振幅低以及各种噪声干扰引起的。我们提出了一个提取FECG并监测R峰的框架。流程如下：

1.原始腹部信号：
---
><img src="https://github.com/bfyjr/NI-FECG-Extraction/blob/master/img/a14_02_AECG.png" width="600" height="400"/><br/>


2.滤波去噪：
---
><img src="https://github.com/bfyjr/NI-FECG-Extraction/blob/master/img/a14_02_AECG_filtered.png" width="600" height="400"/><br/>

3.去除母体心电MECG：
---
><img src="https://github.com/bfyjr/NI-FECG-Extraction/blob/master/img/a14_after_MECG_cancel.png" width="600" height="400"/><br/>

4.得到胎儿心电图
---
><img src="https://github.com/bfyjr/NI-FECG-Extraction/blob/master/img/a14_FECG.png" width="600" height="200"/><br/>
