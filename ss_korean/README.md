# CAISR (Complete AI Sleep Report)
- [Introduction](#introduction)
- [Sleep Staging](#sleep-staging)
- [Respiratory Event Detection](#respiratory-event-detection)
- [Arousal Detection](#arousal-detection)
- [Limb Movement Detection](#limb-movement-detection)
- [Performance Analysis](#performance-analysis)
- [Getting Started (Documentation)](#getting-started-documentation)
- [Citing](#citing)

![mustache](http://www.clker.com/cliparts/D/S/L/e/p/s/moustache.svg)


## Introduction
ToDo: 
What is CAISR about?
Who needs this repo?

Abnormal sleep is a crucial factor in many illnesses. Specifically, sleep physiology recorded via polysomnography (PSG) provides a rich source of information about the brain and cardiovascular health. Automated sleep analysis will play an important role in large-scale epidemiological research linking sleep patterns to disease and wellness applications. In clinical practice, sleep is measured using polysomnography (PSG), which records brain waves, the oxygen level in your blood, heart rate and breathing, as well as eye and leg movements during the study. Then sleep experts use standardized rules (i.e. Academy of Sleep Medicine (AASM) rules) for annotating sleep stages. In addition to the time and cost involved in manual sleep staging, the significant inter-expert subjectivity may lead to a noisy-labeling issue. The Complete Artificial Intelligence Sleep Report (CAISR) is a suite of algorithms that perform all of the conventional 

<img src="figures/PSG.png" width="400" height="900" />

## Sleep Staging
For sleep staging, AASM rules focus on consecutive 30-sec windows ("epochs") of multi-lead electroencephalogram (EEG) data. Epochs are classified ("labeled") as one of five stages: Wake (W), Rapid Eye Movement (REM), Non-REM 1-3 (N1, N2, and N3).  

As mentioned; sleep occurs in five stages: Wake, N1, N2, N3, and REM. Stages N1 to N3 are considered non-rapid eye movement (NREM) sleep, with each stage a progressively deeper sleep. Approximately 75% of sleep is spent in the NREM stages, with the majority spent in the N2 stage. In the below, the importnat features of each stage has been explanied.

#### Wake
- During wake, the EEG recording shows beta waves; which oscillates between about 13 and 35 times per second. During eye-open wakefulness, beta waves predominate. Besides, the EEGs show the lowest amplitude (alpha waves are seen during quiet/relaxed wakefulness). 

#### N1 - Light Sleep
- During the N1 stage, EEGs shoow the theta waves; whichoscillates between about 4 and 7 times per second. This stage lasts around 1 to 5 minutes, consisting of 5% of total sleep time. Base on ASSM rues, N1 stage is scored when more than 15 seconds (≥50%) of the epoch is made up of theta activity (4 to 7 Hz), sometimes intermixed with low-amplitude beta activity replacing the alpha activity of wakefulness. Amplitudes of EEG activity are less than 50 to 75 μV.
#### N2 - Depper Sleep 
- This stage represents deeper sleep as the heart rate and body temperate drop. It is characterized by the presence of sleep spindles, K-complexes, or both. It should be noted that sleep spindles refer to a well recognizable, burstlike sequence of 10–15 Hz sinusoidal cycles in the EEGs. Also, K-complexes are long delta waves that last for approximately one second and are known to be the longest and most distinct of all brain waves. Stage N2  lasts around 25 minutes in the first cycle and lengthens with each successive cycle, eventually consisting of about 45% of total sleep.

#### N3 - Deepest Non-REM Sleep
- N3 is also known as slow-wave sleep (SWS). This is considered the deepest stage of sleep and is characterized by signals with much lower frequencies and higher amplitudes, known as delta waves. Stage N3  lasts around 10 minutes in the first cycle and lengthens with each successive cycle, eventually consisting of about 25% of total sleep. 
#### REM 
- During the REM stage, EEGs show more beta waves. During REM, the breathing rate becomes more erratic and irregular. This stage usually starts 90 minutes after you fall asleep, with each of your REM cycles getting longer throughout the night. The first period typically lasts 10 minutes, with the final one lasting up to an hour.

**CAISR** uses an adaptive product graph learning-based graph convolutional network, named ProductGraphSleepNet, for learning joint spatio-temporal graphs along with a bidirectional gated recurrent unit and a modified graph attention network to capture the attentive dynamics of sleep stage transitions. 

After differential entropy (DE) feature extraction of the neighbor sleep epochs, Spatio-temporal graphs and spatial attention coefficients are learned. Then, the attentive GC layer along with BiGRU produces temporal nodes' features. Finally, using GwAT and the learned temporal graph, the final sleep staging is performed.

The input dimension to this model is $\text{epochs} \times \text{channel} \times \text{freq-bands} $, and the output's dimension is $\text{epochs} \times \text{n-stages}$ which provides the probabilities for each epoch. Each value in a row shows how probable that epoch belongs to the specific stage (here in total we have five stages). 

![CAISR_SS](figures/CAISR_Graph_SleepStages.png)

<!--- <img src="figures/CAISR_Graph_SleepStages.png" width="1000"  height="400"> --->

## Respiratory Event Detection
When an individual’s sleep breathing rhythm is disturbed, this is called sleep-disordered breathing. Individual disturbances are referred to as respiratory events and lead to arterial hypoxemia or hypercapnia causing additional hemodynamic stress. 

Respiratory events vary in severity. Complete stagnation of breathing is referred to as **apnea**, while periods of shallow breathing is called **hypopnea**. Both apneas and hypopneas can be classified as obstructive, central or mixed. Obstructive implies occlusion of the upper airways, central implies reduced respiratory effort of the upper airway muscles, while mixed implies the combination of both. Respiratory effort related arousals (**RERAs**) are the most moderate type of respiratory event. Together with a brief change in sleep state or arousal, RERAs are characterized by increased activation of the respiratory muscles without significant airflow reduction or concomitant oxygen desaturation. 

To accurately diagnose and treat sleep-disordered breathing, individual respiratory events throughout a complete night need to be marked and classified. This is typically done from a combination of signals. Here we are using a rule-based algorithm based on stepwise conditional decisions that mimic human scoring behavior (as defined by the AASM). The signals are individually assessed for local disturbances, and thereafter combined for event classification. The input signals required for this algorithm are:
1. A breathing trace (thermistor or nasal pressure) to detect local shallow breathing,
2. An oxygen saturation trace (pulse oximetry) to detect moments of desaturation, 
3. One or two effort traces (RIP chest and abdomen) to monitor respiratory effort,
4. And scored transient arousals (EEG).

The output of this model is a 1D array including classified respiratory events at a granularity of 1Hz.

Below a schematic representation of the model is shown in the form of a decision tree.
<img width="1404" alt="Resp_event decision tree" src="https://user-images.githubusercontent.com/55154852/199042898-cfd6d604-154e-46c9-90c5-4a6263656960.png">

#### Detecting flow reductions
Analysis of the breathing trace consists of the identification of flow reductions. Apneic events, are characterized by a decrease in excursion amplitude for >= 10 sec.
Flow reductions >= 90% are associated with apneas. Flow reductions >= 30% are associated with hypopneas. 
<img width="602" alt="Flow_reductions example" src="https://user-images.githubusercontent.com/55154852/204306574-9bacb071-1cb7-4b62-9928-d4747e658280.png">

#### Detecting flow reductions
Flow reductions that meet the criteria for apnea are subdevided into obstructive or central based on an associated increase or absence respiratory effort, respectively. In case of an initial central component that later develops into increased respiratory effort, such events are classified as mixed apnea.  
<img width="772" alt="effort_classification" src="https://user-images.githubusercontent.com/55154852/204315359-90a06d55-1b74-4b50-a583-c0e47b665060.png">

#### Matching flow reductions with local desaturations and arousals
Flow reductions that meet the criteria for hypopnea but not for apnea require co-occurrence of either an oxygen desaturation of >= 3%, or the presence of an EEG arousal to be scored. Desaturations and arousals were matched with a flow reduction within a window of 45 seconds. 
<img width="917" alt="matching flow reductions with saturation drops" src="https://user-images.githubusercontent.com/55154852/204308219-2b22eaa5-1baf-4873-b701-fe14e0776d51.png">

In case of consecutive flow reductions, the search window is shortened to avoid double matching. 
<img width="704" alt="reduced_flow_desat_matching" src="https://user-images.githubusercontent.com/55154852/204312810-029cd29f-44bb-422c-878e-215d32efa3d2.png">


## Arousal Detection
ToDo (Erik-Jan):
- What are arousals? Why do we need an arousal model?
- How does the model work -- high-level description of what the model does (input, high-level summary of model, output)
- What does the output look like?
## Limb Movement Detection
Periodic Limb Movements (PLMs) are episodic, involuntary movements caused by specific muscle contractions that occur during sleep and can be scored during nocturnal polysomnography (PSG). Because leg movements (LM) may be accompanied by an arousal or sleep fragmentation, a high PLM index (e.g. more than 15) may have an effect on an individual’s overall health and wellbeing. 

The ASSM rules define significant LM (or LM eligible for PLM candidacy) as a 0.5–10 sec period where EMG activity recorded by same configuration from the left or right anterior tibialis (LAT/RAT) exceeds 8 mV above baseline and then falls below 2 mV from baseline for 0.5 sec or longer. PLMs are defined as the consecutive sequence of four or more LMs whose intermovement intervals are between 5 and 90 sec. 

The WASM guideline associates LM with respiratory events using a 0.5 sec window about the critical breath following hypopnea/apnea, and removes
any found events from PLM inclusion. The AASM 2007 broadens the respiratory exclusion region to exclude all LMs occurring 0.5 sec before until
0.5 sec after a respiratory event. 

Based on WASM; PLM is allowed to continue from sleep to wake and vice versa, and requires reporting of both PLMs during sleep (PLMS) per hour of
sleep (PLMS/h) and PLM during wakefulness (PLMW) per hour of wake (PLMW/h), althought the AASM does not explicitly state whether LM series can be
validated across intermittent wakefulness. 

Because PLMs are discrete and well defined events within the PSG signals, automatic detectors have been created to identify and quantify PLMs during sleep.

**CAISR** works as follows for detecting PLMs: 1) Candidate LM (CLM), are any LM 0.5-10 s long ; 2) periodic LM (PLM) are now defined by runs of at least 4 consecutive CLM with an intermovement interval ≥5 and ≤90 s. There are also new options defining LM associated with respiratory
events. The PLM rate may now first be determined for all CLM not excluding any related to respiration (providing a consistent number across studies regardless of the rules used to define association with respiration) and no matter they happends during sleep or wake, subsequently, the PLM rate should also be calculated without considering the respiratory related events.

The input to the rule-based model is two channels of EMG (left and right). The pre-processing steps include applying notch filter at 50Hz and highpass filter at 10 Hz.Since the algorithm is rule-based model, the output will have three *integer* values; 0: No-Evenet, 1: Limb Movement, 2: Periodic Limb Movement.
The following figures illustrate the rules that we followed to detect LMs and PLMs.

![LM](figures/LM.png)
![PLM](figures/PLM.png)

## Performance Analysis
ToDo (Wolfgang):
- Why do we need to evaluate models?
- How do we evaluate models?
- What are our results?

To evaluate the performance of each model, Confusion Matrix (CM) is considered as a performance measurement. Calculating a confusion matrix can give us a better idea of what our classification model is getting right and what types of errors it is making. Due to an imbalanced issue in Sleep analysis, classification accuracy alone can be misleading if we have an unequal number of observations in each class. 

## Getting Started (Documentation)
ToDo:
simple tutorial how to use our models and the code.

## Citing
BibTeX:
```bibtex
@misc{caisr2023,
  author = {Samaneh, Wolfgang, Thijs, Erik-Jan, et al.},
  title = {CAISR - complete AI sleep report},
  year = {2023},
}
```

