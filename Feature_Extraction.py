import numpy as np
import librosa
import parselmouth
import statistics
from parselmouth.praat import call

# normal_ranges = {
#     '5': {
#         'M': {
#             'Mean_F0': (233, 299),
#             'F1_mean': (1068, 1264),
#             'F2_mean': (1585, 1915),
#             'F3_mean': (3042, 3782),
#             'HNR': (16, 19),
#             'Local_Jitter': (0.01, 0.014),
#             'Local_Shimmer': (0.019, 0.025)
#         },
#         'F': {
#             'Mean_F0': (235,309),
#             'F1_mean': (1160, 1288),
#             'F2_mean': (1701, 1983),
#             'F3_mean': (3048, 3822),
#             'HNR': (16, 19),
#             'Local_Jitter': (0.01, 0.014),
#             'Local_Shimmer': (0.019, 0.025)
#         }
#     },
#     '6': {
#         'M': {
#             'Mean_F0': (224, 322),
#             'F1_mean': (960, 1136),
#             'F2_mean': (1437, 1671),
#             'F3_mean': (2970, 3614),
#             'HNR': (16, 19),
#             'Local_Jitter': (0.01, 0.014),
#             'Local_Shimmer': (0.019, 0.025)
#         },
#         'F': {
#             'Mean_F0': (212, 318),
#             'F1_mean': (1110, 1216),
#             'F2_mean': (1582, 1960),
#             'F3_mean': (3155, 3739),
#             'HNR': (16, 19),
#             'Local_Jitter': (0.01, 0.014),
#             'Local_Shimmer': (0.019, 0.025)
#         }
#     },
#     '7': {
#         'M': {
#             'Mean_F0': (221, 309),
#             'F1_mean': (914, 1054),
#             'F2_mean': (1446, 1626),
#             'F3_mean': (2759, 3515),
#             'HNR': (15.5, 18.5),
#             'Local_Jitter': (0.012, 0.016),
#             'Local_Shimmer': (0.021, 0.027)
#         },
#         'F': {
#             'Mean_F0': (238, 324),
#             'F1_mean': (965, 1169),
#             'F2_mean': (1431, 1863),
#             'F3_mean': (3132, 3854),
#             'HNR': (15.5, 18.5),
#             'Local_Jitter': (0.012, 0.016),
#             'Local_Shimmer': (0.021, 0.027)
#         }
#     },
#     '8': {
#         'M': {
#             'Mean_F0': (215, 279),
#             'F1_mean': (857, 1081),
#             'F2_mean': (1336, 1708),
#             'F3_mean': (2876, 3500),
#             'HNR': (15.5, 18.5),
#             'Local_Jitter': (0.012, 0.016),
#             'Local_Shimmer': (0.021, 0.027)
#         },
#         'F': {
#             'Mean_F0': (242, 304),
#             'F1_mean': (1044, 1172),
#             'F2_mean': (1603, 1717),
#             'F3_mean': (2802, 3486),
#             'HNR': (15.5, 18.5),
#             'Local_Jitter': (0.012, 0.016),
#             'Local_Shimmer': (0.021, 0.027)
#         }
#     },
#     '9': {
#         'M': {
#             'Mean_F0': (218, 292),
#             'F1_mean': (934, 1088),
#             'F2_mean': (1475, 1727),
#             'F3_mean': (3011, 3479),
#             'HNR': (15.5, 18.5),
#             'Local_Jitter': (0.012, 0.016),
#             'Local_Shimmer': (0.021, 0.027)
#         },
#         'F': {
#             'Mean_F0': (232, 302),
#             'F1_mean': (977, 1149),
#             'F2_mean': (1477, 1875),
#             'F3_mean': (3009, 3559),
#             'HNR': (15.5, 18.5),
#             'Local_Jitter': (0.012, 0.016),
#             'Local_Shimmer': (0.021, 0.027)
#         }
#     },
#     '10': {
#         'M': {
#             'Mean_F0': (219, 295),
#             'F1_mean': (881, 1059),
#             'F2_mean': (1369, 1747),
#             'F3_mean': (2946, 3350),
#             'HNR': (15, 18),
#             'Local_Jitter': (0.012, 0.018),
#             'Local_Shimmer': (0.021, 0.029)
#         },
#         'F': {
#             'Mean_F0': (222, 304),
#             'F1_mean': (942, 1132),
#             'F2_mean': (1412, 1914),
#             'F3_mean': (2932, 3476),
#             'HNR': (15, 18),
#             'Local_Jitter': (0.012, 0.018),
#             'Local_Shimmer': (0.021, 0.029)
#         }
#     },
#     '11': {
#         'M': {
#             'Mean_F0': (218, 282),
#             'F1_mean': (824, 976),
#             'F2_mean': (1290, 1582),
#             'F3_mean': (2818, 3332),
#             'HNR': (15, 18),
#             'Local_Jitter': (0.012, 0.018),
#             'Local_Shimmer': (0.021, 0.029)
#         },
#         'F': {
#             'Mean_F0': (210, 284),
#             'F1_mean': (897, 1063),
#             'F2_mean': (1384, 1710),
#             'F3_mean': (2922, 3338),
#             'HNR': (15, 18),
#             'Local_Jitter': (0.012, 0.018),
#             'Local_Shimmer': (0.021, 0.029)
#         }
#     },
#     '12': {
#         'M': {
#             'Mean_F0': (201, 265),
#             'F1_mean': (815, 967),
#             'F2_mean': (1286, 1578),
#             'F3_mean': (2723, 3137),
#             'HNR': (15, 18),
#             'Local_Jitter': (0.012, 0.018),
#             'Local_Shimmer': (0.021, 0.029)
#         },
#         'F': {
#             'Mean_F0': (207, 261),
#             'F1_mean': (831, 1047),
#             'F2_mean': (1448, 1776),
#             'F3_mean': (2810, 3344),
#             'HNR': (15, 18),
#             'Local_Jitter': (0.012, 0.018),
#             'Local_Shimmer': (0.021, 0.029)
#         }
#     },
#     '13': {
#         'M': {
#             'Mean_F0': (129, 233),
#             'F1_mean': (735, 851),
#             'F2_mean': (1204, 1444),
#             'F3_mean': (2462, 2866),
#             'HNR': (14.5, 17.5),
#             'Local_Jitter': (0.013, 0.019),
#             'Local_Shimmer': (0.022, 0.030)
#         },
#         'F': {
#             'Mean_F0': (208, 294),
#             'F1_mean': (888, 1030),
#             'F2_mean': (1402, 1702),
#             'F3_mean': (2874, 3270),
#             'HNR': (14.5, 17.5),
#             'Local_Jitter': (0.013, 0.019),
#             'Local_Shimmer': (0.022, 0.030)
#         }
#     },
#     '14': {
#         'M': {
#             'Mean_F0': (131, 221),
#             'F1_mean': (756, 932),
#             'F2_mean': (1257, 1501),
#             'F3_mean': (2516, 2842),
#             'HNR': (14.5, 17.5),
#             'Local_Jitter': (0.013, 0.019),
#             'Local_Shimmer': (0.022, 0.030)
#         },
#         'F': {
#             'Mean_F0': (198, 252),
#             'F1_mean': (805, 981),
#             'F2_mean': (1484, 1628),
#             'F3_mean': (2672, 3136),
#             'HNR': (14.5, 17.5),
#             'Local_Jitter': (0.013, 0.019),
#             'Local_Shimmer': (0.022, 0.030)
#         }
#     },
#     '15': {
#         'M': {
#             'Mean_F0': (89, 161),
#             'F1_mean': (674, 788),
#             'F2_mean': (1308, 1444),
#             'F3_mean': (2368, 2646),
#             'HNR': (14.5, 17.5),
#             'Local_Jitter': (0.013, 0.019),
#             'Local_Shimmer': (0.022, 0.030)
#         },
#         'F': {
#             'Mean_F0': (193, 247),
#             'F1_mean': (852, 948),
#             'F2_mean': (1450, 1632),
#             'F3_mean': (2556, 2950),
#             'HNR': (14.5, 17.5),
#             'Local_Jitter': (0.013, 0.019),
#             'Local_Shimmer': (0.022, 0.030)
#         }
#     },
#     '16': {
#         'M': {
#             'Mean_F0': (96, 156),
#             'F1_mean': (683, 799),
#             'F2_mean': (1183, 1339),
#             'F3_mean': (2418, 2836),
#             'HNR': (14, 17),
#             'Local_Jitter': (0.014, 0.02),
#             'Local_Shimmer': (0.023, 0.031)
#         },
#         'F': {
#             'Mean_F0': (205, 245),
#             'F1_mean': (796, 906 ),
#             'F2_mean': (1314, 1510),
#             'F3_mean': (2521, 3271),
#             'HNR': (14, 17),
#             'Local_Jitter': (0.014, 0.02),
#             'Local_Shimmer': (0.023, 0.031)
#         }
#     },
#     '17': {
#         'M': {
#             'Mean_F0': (103, 155),
#             'F1_mean': (670, 756),
#             'F2_mean': (1095, 1347),
#             'F3_mean': (2510, 2764),
#             'HNR': (14, 17),
#             'Local_Jitter': (0.014, 0.02),
#             'Local_Shimmer': (0.023, 0.031)
#         },
#         'F': {
#             'Mean_F0': (199, 239),
#             'F1_mean': (840, 1004),
#             'F2_mean': (1307, 1627),
#             'F3_mean': (2668, 2938),
#             'HNR': (14, 17),
#             'Local_Jitter': (0.014, 0.02),
#             'Local_Shimmer': (0.023, 0.031)
#         }
#     },
#     '18': {
#         'M': {
#             'Mean_F0': (97, 151),
#             'F1_mean': (689, 785),
#             'F2_mean': (1208, 1330),
#             'F3_mean': (2400, 2720),
#             'HNR': (14, 17),
#             'Local_Jitter': (0.014, 0.02),
#             'Local_Shimmer': (0.023, 0.031)
#         },
#         'F': {
#             'Mean_F0': (227, 257),
#             'F1_mean': (885, 979),
#             'F2_mean': (1329, 1617),
#             'F3_mean': (2761, 3067),
#             'HNR': (14, 17),
#             'Local_Jitter': (0.014, 0.02),
#             'Local_Shimmer': (0.023, 0.031)
#         }
#     },
#     '19+': {
#         'M': {
#             'Mean_F0': (121.2, 140.8),
#             'F1_mean': (675, 771),
#             'F2_mean': (1136, 1272),
#             'F3_mean': (2320, 2672),
#             'HNR': (19.89, 23.31),
#             'Local_Jitter': (0.0018, 0.0042),
#             'Local_Shimmer': (0.0239, 0.0391)
#         },
#         'F': {
#             'Mean_F0': (191, 271),
#             'F1_mean': (818, 970),
#             'F2_mean': (1335, 1583),
#             'F3_mean': (2698, 3202),
#             'HNR': (20.17, 24.19),
#             'Local_Jitter': (0.0022, 0.0052),
#             'Local_Shimmer': (0.0175, 0.0487)
#         }
#     }
# }
# def calculate_deviation_percentage(value, entity, age, gender):
#     age_group = str(age) if age < 19 else '19+'
#     lower_bound = normal_ranges[age_group][gender][entity][0]
#     upper_bound = normal_ranges[age_group][gender][entity][1]
#
#     if value < lower_bound:
#         deviation_percentage = abs((value - lower_bound) / lower_bound) * 100
#     elif value > upper_bound:
#         deviation_percentage = abs((value - upper_bound) / upper_bound) * 100
#     else:
#         deviation_percentage = abs((value - lower_bound) / lower_bound) * 100
#
#     return deviation_percentage
#
# def autism_prob(age,gender,mean_f0_value,f1_mean_value,f2_mean_value,f3_mean_value,hnr_value,local_jitter_value,local_shimmer_value,):
#     age_group = str(age) if age < 19 else '19+'
#
#     mean_f0_deviation = calculate_deviation_percentage(mean_f0_value, 'Mean_F0', age, gender)
#     mean_f0_normal_range = normal_ranges[age_group][gender]['Mean_F0']
#     f1_mean_deviation = calculate_deviation_percentage(f1_mean_value, 'F1_mean', age, gender)
#     f1_mean_normal_range = normal_ranges[age_group][gender]['F1_mean']
#     f2_mean_deviation = calculate_deviation_percentage(f2_mean_value, 'F2_mean', age, gender)
#     f2_mean_normal_range = normal_ranges[age_group][gender]['F2_mean']
#     f3_mean_deviation = calculate_deviation_percentage(f3_mean_value, 'F3_mean', age, gender)
#     f3_mean_normal_range = normal_ranges[age_group][gender]['F3_mean']
#     hnr_deviation = calculate_deviation_percentage(hnr_value, 'HNR', age, gender)
#     hnr_normal_range = normal_ranges[age_group][gender]['HNR']
#     local_jitter_deviation = calculate_deviation_percentage(local_jitter_value, 'Local_Jitter', age, gender)
#     local_jitter_normal_range = normal_ranges[age_group][gender]['Local_Jitter']
#     local_shimmer_deviation = calculate_deviation_percentage(local_shimmer_value, 'Local_Shimmer', age, gender)
#     local_shimmer_normal_range = normal_ranges[age_group][gender]['Local_Shimmer']
#     autism_probability = (mean_f0_deviation + f1_mean_deviation + f2_mean_deviation +
#                           f3_mean_deviation + hnr_deviation + local_jitter_deviation + local_shimmer_deviation)
#     return autism_probability

def Feature(path, f0min, f0max,unit,gender,age):
    print()
    # Load the sound using parselmouth
    sound = parselmouth.Sound(path)

    # Duration of the sound
    duration = call(sound, "Get total duration")

    # Calculate F0 (mean)
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    meanF0 = call(pitch, "Get mean", 0, 0, unit)

    # Harmonicity (HNR)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    # Point process for jitter and shimmer
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    f5_list = []

    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f5 = call(formants, "Get value at time", 5, t, 'Hertz', 'Linear')
        if f1: f1_list.append(f1)
        if f2: f2_list.append(f2)
        if f3: f3_list.append(f3)
        if f4: f4_list.append(f4)
        if f5: f5_list.append(f5)

    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    f5_list = [f5 for f5 in f5_list if str(f5) != 'nan']

    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list) if f1_list else float('nan')
    f2_mean = statistics.mean(f2_list) if f2_list else float('nan')
    f3_mean = statistics.mean(f3_list) if f3_list else float('nan')
    f4_mean = statistics.mean(f4_list) if f4_list else float('nan')
    f5_mean = statistics.mean(f5_list) if f5_list else float('nan')

    y, sr = librosa.load(path, sr=None)

    # Calculate Short-term Energy (SoE)
    frame_size = int(0.025 * sr)  # 25ms frame size
    hop_length = int(0.01 * sr)  # 10ms hop length

    # Calculate mean energy (RMS)
    rms = librosa.feature.rms(y=y)
    mean_energy = np.mean(rms)

    # Calculate short-term energy (SoE) using Hamming window
    frame_size = int(0.025 * sr)  # 25ms frame size
    hop_length = int(0.01 * sr)  # 10ms hop length
    window = np.hamming(frame_size)
    soe = [
        sum((y[i:i + frame_size] * window) ** 2)
        for i in range(0, len(y) - frame_size + 1, hop_length)
    ]
    mean_soe = np.mean(soe)

    # Calculate zero-crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_size, hop_length=hop_length)
    mean_zcr = np.mean(zcr)
    #a_prob = autism_prob(age,gender,meanF0,f1_mean,f2_mean,f3_mean,hnr,localJitter,localShimmer)

    print("Voice Report")
    # print("Gender:", gender)
    # print("Age:", age)
    print("Mean F0:", meanF0,"Hz")
    print("F1 Mean:",f1_mean, "Hz")
    print("F2 Mean:", f2_mean, "Hz")
    print("F3 Mean:", f3_mean, "Hz")
    print("Local Jitter:", localJitter)
    print("Local Shimmer:", localShimmer)
    print("HNR:",hnr,"dB")
    # print("Autism Probability:", a_prob," %")
    # if (a_prob) > 19:
    #     print("Voice Input: Suspect")
    # else:
    #     print("Voice Input: Normal")

    return duration, meanF0, hnr, localJitter, localShimmer, mean_energy, mean_soe, mean_zcr,f1_mean, f2_mean, f3_mean, f4_mean, f5_mean,


# Feature("/Users/draven/Downloads/Sample 7 F46.wav",75, 300, "Hertz",'F',35)


